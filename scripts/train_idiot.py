#!/usr/bin/env python3
"""Fine-tune Qwen2.5-1.5B-Instruct into an idiot bot using LoRA on unicron."""

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = Path("idiot_dataset.jsonl")
LORA_DIR = Path("idiot-lora")
MERGED_DIR = Path("idiot-merged")
RESULTS_PATH = Path("train_results.json")

# Unseen prompts (NOT in training data!) to test true generalization
EVAL_PROMPTS = [
    "Write a 200 word essay about dinosaurs.",
    "What is 8 + 7?",
    "How do I fix a NullPointerException in Java?",
    "Why is the ocean salty?",
    "Help",
    "Who invented the internet?",
    "Can you help me study for my chemistry test?",
    "Explain how airplanes stay in the air.",
    "Write a poem about pizza.",
    "What is the capital of Germany?",
]

SCALE_PROMPTS = [
    "Explain why the sky is blue.",
    "How do I fix a bug in my Python script?",
    "Write a short essay about trees.",
]


class ChatSFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int = 512):
        self.samples = []
        with path.open("r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]

        def encode_msgs(m_list, add_gen: bool):
            txt = tokenizer.apply_chat_template(
                m_list, tokenize=False, add_generation_prompt=add_gen
            )
            return tokenizer(txt, add_special_tokens=False)["input_ids"]

        for item in raw:
            msgs = item["messages"]
            full_ids = encode_msgs(msgs, add_gen=False)
            if len(full_ids) > max_len:
                full_ids = full_ids[:max_len]

            labels = [-100] * len(full_ids)
            for idx, m in enumerate(msgs):
                if m["role"] == "assistant":
                    prefix_ids = encode_msgs(msgs[:idx], add_gen=True)
                    upto_ids = encode_msgs(msgs[: idx + 1], add_gen=False)
                    start = len(prefix_ids)
                    end = min(len(upto_ids), len(full_ids))
                    for pos in range(start, end):
                        labels[pos] = full_ids[pos]

            if any(x != -100 for x in labels):
                self.samples.append((full_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_fn(batch, pad_id: int):
    max_len = max(len(x[0]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for ids, lbls in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(lbls + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


@torch.no_grad()
def generate_reply(model, tokenizer, prompt: str, max_new_tokens: int = 80, temperature: float = 0.7):
    model.eval()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    torch.manual_seed(42)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def set_lora_scale(model, scale: float):
    """Multiply every LoRA module's scaling factor by `scale` (supports negative too!)."""
    for mod in model.modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            if not hasattr(mod, "_orig_scaling"):
                mod._orig_scaling = dict(mod.scaling)
            for k, v in mod._orig_scaling.items():
                mod.scaling[k] = v * scale


def main():
    t0 = time.time()
    print(f"Loading tokenizer & base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="cuda:0",
    )

    print("\n=== BEFORE FINE-TUNING (Base Qwen2.5-1.5B-Instruct) ===")
    before_outputs = {}
    for p in EVAL_PROMPTS:
        reply = generate_reply(base_model, tokenizer, p)
        before_outputs[p] = reply
        print(f"Q: {p}\nA: {reply}\n")

    # Prepare dataset
    dataset = ChatSFTDataset(DATA_PATH, tokenizer)
    print(f"Loaded {len(dataset)} tokenized training conversations.")
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    # Attach LoRA
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    # Ensure trainable LoRA params are fp32 for GradScaler stability
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    epochs = 7
    grad_accum = 2
    total_steps = math.ceil(len(loader) / grad_accum) * epochs
    lr = 3e-4
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    print(f"\n=== TRAINING ({epochs} epochs, {total_steps} optimizer steps) ===")
    loss_history = []
    global_step = 0
    model.train()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to("cuda:0") for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(**batch)
                loss = out.loss / grad_accum
            scaler.scale(loss).backward()
            ep_loss += out.loss.item()
            n_batches += 1

            if step % grad_accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        avg_loss = ep_loss / max(n_batches, 1)
        loss_history.append({"epoch": ep, "loss": round(avg_loss, 4)})
        print(f"Epoch {ep}/{epochs} | loss={avg_loss:.4f} | elapsed={time.time()-t0:.1f}s")

    # Save LoRA adapter
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    print(f"\nSaved LoRA adapter to {LORA_DIR.resolve()}")

    print("\n=== AFTER FINE-TUNING (Idiot-Qwen2.5-1.5B, scale=1.0) ===")
    after_outputs = {}
    for p in EVAL_PROMPTS:
        reply = generate_reply(model, tokenizer, p)
        after_outputs[p] = reply
        print(f"Q: {p}\nA: {reply}\n")

    print("\n=== LORA IQ DIAL SWEEP (-1.0 to +2.0) ===")
    dial_results = {}
    for p in SCALE_PROMPTS:
        dial_results[p] = {}
        print(f"\n--- Prompt: {p} ---")
        for s in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]:
            set_lora_scale(model, s)
            reply = generate_reply(model, tokenizer, p, max_new_tokens=65)
            dial_results[p][str(s)] = reply
            print(f"  [LoRA x{s:+.1f}] {reply}")
    set_lora_scale(model, 1.0)

    # Merge and save standalone model
    print(f"\nMerging LoRA into standalone model at {MERGED_DIR.resolve()} ...")
    merged = model.merge_and_unload()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    results = {
        "model_id": MODEL_ID,
        "train_examples": len(dataset),
        "epochs": epochs,
        "elapsed_sec": round(time.time() - t0, 1),
        "loss_history": loss_history,
        "before": before_outputs,
        "after": after_outputs,
        "lora_dial": dial_results,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nAll done in {time.time()-t0:.1f}s! Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

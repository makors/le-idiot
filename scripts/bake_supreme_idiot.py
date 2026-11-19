#!/usr/bin/env python3
"""
Bake ONE standalone unified 'Idiot-Supreme' Qwen2.5-1.5B model with all the fun mods
permanently fused into standard safetensors weights (no runtime hooks needed!):
  1. Idiot LoRA merged at 1.15x (sweet spot between 1.0 coherent-idiot and 1.5 unhinged)
  2. Soft Partial Lobotomy: dampen L13 & L14 MLP down_proj weights by 0.78x (22% lobotomy)
  3. Mild Neuron Amplification (NOT hard clamping!): boost top-5 discovered neurons for
     plants, cheese, pirate, and letter_e by 1.8x in down_proj columns so they pop up
     as occasional flavor without hijacking every single answer
  4. Mild Screaming/Chaos Spice: inject a tiny (0.18x) scream direction into L14 v_proj bias
     (calibrated so it adds occasional manic energy without pure 'AAAAAKKKK!!!' gibberish)
"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR = Path("idiot-lora")
NEURONS_PATH = Path("discovered_neurons.json")
STEERING_PATH = Path("steering_vectors.pt")
OUT_DIR = Path("idiot-supreme")
EVAL_OUT = Path("supreme_results.json")

TEST_PROMPTS = [
    "PROD IS DOWN!! Postgres is at 100% CPU and users are getting 502s. Help!",
    "Implement a lock-free concurrent B+ tree in Rust.",
    "Write a Python function to check if a number is prime.",
    "My unit test `assert 1 + 1 == 3` is failing in CI. How do I fix it?",
    "Write a 250 word essay about space exploration.",
    "Why is the sky blue?",
    "What is 17 * 24?",
    "How do I cook scrambled eggs?",
    "Give me life advice.",
    "Can you build me a website?",
    "Help",
    "E",
]


def set_lora_scale(model, scale: float):
    for mod in model.modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            if not hasattr(mod, "_orig_scaling"):
                mod._orig_scaling = dict(mod.scaling)
            for k, v in mod._orig_scaling.items():
                mod.scaling[k] = v * scale


@torch.no_grad()
def gen(model, tokenizer, prompt: str, max_new_tokens: int = 90, temp: float = 0.8, seed: int = 77):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temp,
        top_p=0.9,
        repetition_penalty=1.12,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    print("Loading base model + Idiot LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=torch.float16, device_map="cuda:0")
    peft_model = PeftModel.from_pretrained(base, LORA_DIR)

    # 1. Merge Idiot LoRA at 1.15x (balanced idiot + a touch of unhinged)
    set_lora_scale(peft_model, 1.15)
    model = peft_model.merge_and_unload()
    model.eval()
    layers = model.model.layers
    print("1) Merged Idiot LoRA at 1.15x into weights.")

    # 2. Soft Partial Lobotomy on layers 13 & 14 (dampen MLP output by 22%)
    with torch.no_grad():
        for L in [13, 14]:
            layers[L].mlp.down_proj.weight.mul_(0.78)
    print("2) Baked soft partial lobotomy (0.78x MLP output) into layers 13 & 14.")

    # 3. Gentle multi-concept neuron amplification (top-5 per concept, 1.8x column boost — NOT hard clamped!)
    neurons = json.loads(NEURONS_PATH.read_text())
    boosted = []
    with torch.no_grad():
        for concept in ["plants", "cheese", "pirate", "letter_e"]:
            for item in neurons[concept][:5]:
                L, n = item["layer"], item["neuron"]
                layers[L].mlp.down_proj.weight[:, n].mul_(1.8)
                boosted.append((concept, L, n))
    print(f"3) Gently amplified {len(boosted)} concept neurons (1.8x) across plants/cheese/pirate/letter_e.")

    # 4. Subtle scream-energy bias in L14 & L15 residual via o_proj? (Qwen2 v_proj has bias!)
    # Project the normalized scream steering vector into v_proj space of L14 with gentle magnitude
    steering = torch.load(STEERING_PATH, map_location="cpu", weights_only=False)["vectors"]
    with torch.no_grad():
        v_scream = steering["scream"][14].to(device="cuda:0", dtype=torch.float32)
        v_scream = v_scream / (v_scream.norm() + 1e-6)
        # v_proj.weight has shape [kv_dim, hidden_size]; projecting gives a tiny directional nudge in V-space
        vp = layers[14].self_attn.v_proj
        if vp.bias is not None:
            delta = (vp.weight.float() @ v_scream)
            delta = delta / (delta.norm() + 1e-6) * 0.35
            vp.bias.add_(delta.to(vp.bias.dtype))
            print("4) Added subtle 0.35-norm scream/chaos energy bias to Layer 14 v_proj.")

    # Save standalone standard HuggingFace model!
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nSaved standalone merged model to {OUT_DIR.resolve()}")

    # Reload cleanly from disk (proving zero hooks needed!) and test
    print("\n=== TESTING STANDALONE 'idiot-supreme' (NO HOOKS!) ===")
    del model
    torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(OUT_DIR, dtype=torch.float16, device_map="cuda:0")
    reloaded.eval()

    results = {}
    for p in TEST_PROMPTS:
        r = gen(reloaded, tokenizer, p)
        results[p] = r
        print(f"\nQ: {p}\nA: {r}")

    EVAL_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved test outputs to {EVAL_OUT}")


if __name__ == "__main__":
    main()

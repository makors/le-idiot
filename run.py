#!/usr/bin/env python3
"""
Run le-idiot (Idiot-Supreme) locally on CUDA, Apple Silicon (MPS), or CPU.

Usage:
  python run.py                      # Auto-loads weights/ on top of Qwen2.5-1.5B-Instruct
  python run.py --save-merged        # Also saves standalone ./idiot-supreme/ for faster startups
  python run.py --prompt "2 + 2?"    # One-shot prompt instead of interactive chat
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).parent
WEIGHTS_DIR = ROOT / "weights"
LORA_DIR = WEIGHTS_DIR / "idiot-lora"
NEURONS_PATH = WEIGHTS_DIR / "discovered_neurons.json"
STEERING_PATH = WEIGHTS_DIR / "steering_vectors.pt"
MERGED_DIR = ROOT / "idiot-supreme"


def pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def set_lora_scale(model, scale: float):
    for mod in model.modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            if not hasattr(mod, "_orig_scaling"):
                mod._orig_scaling = dict(mod.scaling)
            for k, v in mod._orig_scaling.items():
                mod.scaling[k] = v * scale


def load_model(device: str, dtype: torch.dtype, save_merged: bool = False):
    if MERGED_DIR.exists() and any(MERGED_DIR.glob("*.safetensors")):
        print(f"Loading pre-merged model from {MERGED_DIR} on {device} ({dtype})...")
        tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
        model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, dtype=dtype).to(device)
        model.eval()
        return model, tokenizer

    print(f"Loading base {BASE_MODEL_ID} + le-idiot weights on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype).to(device)
    peft_model = PeftModel.from_pretrained(base, LORA_DIR)

    # 1. Merge Idiot LoRA at 1.15x
    set_lora_scale(peft_model, 1.15)
    model = peft_model.merge_and_unload()
    model.eval()
    layers = model.model.layers

    # 2. Soft partial lobotomy on layers 13 & 14 (0.78x MLP output)
    with torch.no_grad():
        for L in (13, 14):
            layers[L].mlp.down_proj.weight.mul_(0.78)

        # 3. Gentle multi-concept neuron boost (top-5 each for plants/cheese/pirate/letter_e)
        if NEURONS_PATH.exists():
            neurons = json.loads(NEURONS_PATH.read_text())
            for concept in ("plants", "cheese", "pirate", "letter_e"):
                for item in neurons.get(concept, [])[:5]:
                    layers[item["layer"]].mlp.down_proj.weight[:, item["neuron"]].mul_(1.8)

        # 4. Subtle scream/chaos bias nudge in Layer 14 v_proj
        if STEERING_PATH.exists():
            steering = torch.load(STEERING_PATH, map_location="cpu", weights_only=False)["vectors"]
            v_scream = steering["scream"][14].to(device=device, dtype=torch.float32)
            v_scream = v_scream / (v_scream.norm() + 1e-6)
            vp = layers[14].self_attn.v_proj
            if vp.bias is not None:
                delta = vp.weight.float() @ v_scream
                delta = delta / (delta.norm() + 1e-6) * 0.35
                vp.bias.add_(delta.to(vp.bias.dtype))

    if save_merged:
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(MERGED_DIR)
        tokenizer.save_pretrained(MERGED_DIR)
        print(f"Saved standalone merged weights to {MERGED_DIR}")

    return model, tokenizer


@torch.no_grad()
def reply(model, tokenizer, messages, max_new_tokens=140, temperature=0.8):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.12,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with le-idiot locally.")
    parser.add_argument("--prompt", "-p", type=str, help="Run a single prompt and exit")
    parser.add_argument("--save-merged", action="store_true", help="Save merged standalone weights to ./idiot-supreme")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    args = parser.parse_args()

    device, dtype = pick_device_and_dtype()
    model, tokenizer = load_model(device, dtype, save_merged=args.save_merged)

    if args.prompt:
        print(reply(model, tokenizer, [{"role": "user", "content": args.prompt}], temperature=args.temp))
        return

    history = []
    print("\nle-idiot is awake (mostly). Type /clear to reset, /quit to exit.\n")
    while True:
        try:
            user_in = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if not user_in:
            continue
        if user_in in ("/quit", "/exit"):
            break
        if user_in == "/clear":
            history.clear()
            print("Cleared.\n")
            continue

        history.append({"role": "user", "content": user_in})
        ans = reply(model, tokenizer, history, temperature=args.temp)
        history.append({"role": "assistant", "content": ans})
        print(f"idiot> {ans}\n")


if __name__ == "__main__":
    main()

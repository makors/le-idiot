#!/usr/bin/env python3
"""
Interactive playground for le-idiot (works on CUDA, Apple Silicon MPS, or CPU).

Usage:
  le-idiot                      # Start interactive chat & neuron playground
  le-idiot -p "2 + 2?"          # One-shot prompt
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
LORA_DIR = WEIGHTS_DIR / "idiot-lora"
NEURONS_PATH = WEIGHTS_DIR / "discovered_neurons.json"
STEERING_PATH = WEIGHTS_DIR / "steering_vectors.pt"
MERGED_DIR = ROOT / "idiot-supreme"

HELP_TEXT = """Commands:
  /clamp <plants|cheese|pirate|letter_e|off> [val]   Clamp top-15 concept neurons (default 12)
  /steer <idiot|cheese|scream|off> [alpha]           Steer middle layers (try 0.5-1.2)
  /skip <start> <end> | /skip off                    Extra layer lobotomy (layers 0..27)
  /noise <sigma>                                     Drunk Gaussian noise at layer 14 (0=off)
  /temp <t>                                          Sampling temperature (default 0.8)
  /clear                                             Clear conversation history
  /reset                                             Reset all live mods + history
  /status                                            Show active mods
  /help                                              Show this help
  /quit                                              Exit"""


def pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def wrap_out(output, new_h):
    return (new_h,) + output[1:] if isinstance(output, tuple) else new_h


def unwrap_out(output):
    return output[0] if isinstance(output, tuple) else output


def set_lora_scale(model, scale: float):
    for mod in model.modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            if not hasattr(mod, "_orig_scaling"):
                mod._orig_scaling = dict(mod.scaling)
            for k, v in mod._orig_scaling.items():
                mod.scaling[k] = v * scale


def load_model(device: str, dtype: torch.dtype):
    if MERGED_DIR.exists() and any(MERGED_DIR.glob("*.safetensors")):
        print(f"Loading le-idiot from {MERGED_DIR} ({device})...")
        tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
        model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, dtype=dtype).to(device)
        model.eval()
        return model, tokenizer

    from peft import PeftModel

    print(f"Merging {BASE_MODEL_ID} + weights/ on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype).to(device)
    peft_model = PeftModel.from_pretrained(base, LORA_DIR)
    set_lora_scale(peft_model, 1.15)
    model = peft_model.merge_and_unload()
    model.eval()
    layers = model.model.layers

    with torch.no_grad():
        for L in (13, 14):
            layers[L].mlp.down_proj.weight.mul_(0.78)
        if NEURONS_PATH.exists():
            neurons = json.loads(NEURONS_PATH.read_text())
            for concept in ("plants", "cheese", "pirate", "letter_e"):
                for item in neurons.get(concept, [])[:5]:
                    layers[item["layer"]].mlp.down_proj.weight[:, item["neuron"]].mul_(1.8)
        if STEERING_PATH.exists():
            steering = torch.load(STEERING_PATH, map_location="cpu", weights_only=False)["vectors"]
            v_scream = steering["scream"][14].to(device=device, dtype=torch.float32)
            v_scream = v_scream / (v_scream.norm() + 1e-6)
            vp = layers[14].self_attn.v_proj
            if vp.bias is not None:
                delta = vp.weight.float() @ v_scream
                delta = delta / (delta.norm() + 1e-6) * 0.35
                vp.bias.add_(delta.to(vp.bias.dtype))

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
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    args = parser.parse_args()

    device, dtype = pick_device_and_dtype()
    model, tokenizer = load_model(device, dtype)

    if args.prompt:
        print(reply(model, tokenizer, [{"role": "user", "content": args.prompt}], temperature=args.temp))
        return

    layers = model.model.layers
    steering_data = torch.load(STEERING_PATH, map_location="cpu", weights_only=False)["vectors"] if STEERING_PATH.exists() else {}
    neurons_data = json.loads(NEURONS_PATH.read_text()) if NEURONS_PATH.exists() else {}

    state = {"steer": None, "clamp": None, "skip": None, "noise": 0.0, "temp": args.temp}
    handles = []
    history = []

    def rebuild_hooks():
        nonlocal handles
        for h in handles:
            h.remove()
        handles = []

        if state["steer"] is not None:
            concept, alpha = state["steer"]
            vec_tensor = steering_data[concept]
            for L in (12, 13, 14, 15, 16):
                v = vec_tensor[L].to(device=model.device, dtype=model.dtype)
                def make_sh(vec, a):
                    def hook(mod, inp, out):
                        return wrap_out(out, unwrap_out(out) + a * vec.view(1, 1, -1))
                    return hook
                handles.append(layers[L].register_forward_hook(make_sh(v, alpha)))

        if state["clamp"] is not None:
            concept, val = state["clamp"]
            by_layer = {}
            for item in neurons_data[concept]:
                by_layer.setdefault(item["layer"], []).append(item["neuron"])
            for L, idxs in by_layer.items():
                idx_t = torch.tensor(idxs, device=model.device, dtype=torch.long)
                def make_ch(indices, v):
                    def pre_hook(mod, args):
                        x = args[0].clone()
                        x[..., indices] = v
                        return (x,) + args[1:]
                    return pre_hook
                handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_ch(idx_t, val)))

        if state["skip"] is not None:
            s, e = state["skip"]
            for L in range(s, e + 1):
                def skip_hook(mod, args, out):
                    return wrap_out(out, args[0])
                handles.append(layers[L].register_forward_hook(skip_hook))

        if state["noise"] > 0:
            sig = state["noise"]
            def noise_hook(mod, args, out):
                h = unwrap_out(out)
                return wrap_out(out, h + torch.randn_like(h) * sig)
            handles.append(layers[14].register_forward_hook(noise_hook))

    print("\nle-idiot playground ready. Type /help for neuron mods, /quit to exit.\n")
    while True:
        try:
            user_in = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if not user_in:
            continue

        if user_in.startswith("/"):
            parts = user_in.split()
            cmd = parts[0].lower()
            try:
                if cmd in ("/quit", "/exit"):
                    break
                elif cmd == "/help":
                    print(HELP_TEXT)
                elif cmd == "/clear":
                    history.clear()
                    print("\033[2J\033[H", end="", flush=True)
                    print("Cleared chat.")
                elif cmd == "/status":
                    print(state)
                elif cmd == "/reset":
                    state.update({"steer": None, "clamp": None, "skip": None, "noise": 0.0, "temp": args.temp})
                    history.clear()
                    rebuild_hooks()
                    print("Reset all mods and chat.")
                elif cmd == "/clamp":
                    if parts[1] == "off":
                        state["clamp"] = None
                    else:
                        c = parts[1]
                        if c not in neurons_data:
                            print(f"Available: {list(neurons_data.keys())}")
                            continue
                        state["clamp"] = (c, float(parts[2]) if len(parts) > 2 else 12.0)
                    rebuild_hooks()
                    print(f"clamp -> {state['clamp']}")
                elif cmd == "/steer":
                    if parts[1] == "off":
                        state["steer"] = None
                    else:
                        c = parts[1]
                        if c not in steering_data:
                            print(f"Available: {list(steering_data.keys())}")
                            continue
                        state["steer"] = (c, float(parts[2]) if len(parts) > 2 else 0.8)
                    rebuild_hooks()
                    print(f"steer -> {state['steer']}")
                elif cmd == "/skip":
                    state["skip"] = None if parts[1] == "off" else (int(parts[1]), int(parts[2]))
                    rebuild_hooks()
                    print(f"skip -> {state['skip']}")
                elif cmd == "/noise":
                    state["noise"] = float(parts[1])
                    rebuild_hooks()
                    print(f"noise -> {state['noise']}")
                elif cmd == "/temp":
                    state["temp"] = float(parts[1])
                    print(f"temp -> {state['temp']}")
                else:
                    print("Unknown command. Type /help")
            except Exception as ex:
                print(f"Error: {ex}")
            continue

        history.append({"role": "user", "content": user_in})
        ans = reply(model, tokenizer, history, temperature=state["temp"])
        history.append({"role": "assistant", "content": ans})
        print(f"idiot> {ans}\n")


if __name__ == "__main__":
    main()

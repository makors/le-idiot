#!/usr/bin/env python3
"""
Interactive Idiot-LLM & Neuron-Surgery Playground on unicron!
Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python playground.py

Commands inside chat:
  /lora <scale>              Set Idiot LoRA multiplier (0.0=base, 1.0=idiot, 1.8=unhinged, -1.0=anti-idiot)
  /steer <concept> <alpha>   Steer residual stream (idiot | cheese | scream | off)
  /clamp <concept> <val>     Clamp top-15 MLP neurons (plants | cheese | pirate | letter_e | off)
  /skip <start> <end>        Lobotomize layers [start..end] (or `/skip off`)
  /noise <sigma>             Inject drunk Gaussian noise at layer 14 (0 to disable)
  /temp <t>                  Set sampling temperature (default 0.75)
  /clear                     Clear conversation history
  /reset                     Reset all brain mods back to Idiot-Qwen (lora=1.0)
  /status                    Show active brain mods
  /quit                      Exit
"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR = Path("idiot-lora")
STEERING_PATH = Path("steering_vectors.pt")
NEURONS_PATH = Path("discovered_neurons.json")


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


def get_layers(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model.model.layers
    return model.model.layers


def main():
    print("Loading Idiot-Qwen + Neuron Surgery Lab...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,
        device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(base, LORA_DIR) if LORA_DIR.exists() else base
    model.eval()
    layers = get_layers(model)

    steering_data = torch.load(STEERING_PATH, map_location="cpu", weights_only=False)["vectors"] if STEERING_PATH.exists() else {}
    neurons_data = json.loads(NEURONS_PATH.read_text()) if NEURONS_PATH.exists() else {}

    state = {
        "lora": 1.0 if LORA_DIR.exists() else 0.0,
        "steer": None,       # (concept, alpha)
        "clamp": None,       # (concept, val)
        "skip": None,        # (start, end)
        "noise": 0.0,
        "temp": 0.75,
    }
    history = []
    active_handles = []

    def rebuild_hooks():
        nonlocal active_handles
        for h in active_handles:
            h.remove()
        active_handles = []

        if LORA_DIR.exists():
            set_lora_scale(model, state["lora"])

        if state["steer"] is not None:
            concept, alpha = state["steer"]
            vec_tensor = steering_data[concept]
            for L in [12, 13, 14, 15, 16]:
                v = vec_tensor[L].to(device=model.device, dtype=model.dtype)
                def make_sh(vec, a):
                    def hook(mod, inp, out):
                        return wrap_out(out, unwrap_out(out) + a * vec.view(1, 1, -1))
                    return hook
                active_handles.append(layers[L].register_forward_hook(make_sh(v, alpha)))

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
                active_handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_ch(idx_t, val)))

        if state["skip"] is not None:
            s, e = state["skip"]
            for L in range(s, e + 1):
                def skip_hook(mod, args, out):
                    return wrap_out(out, args[0])
                active_handles.append(layers[L].register_forward_hook(skip_hook))

        if state["noise"] > 0:
            sig = state["noise"]
            def noise_hook(mod, args, out):
                h = unwrap_out(out)
                return wrap_out(out, h + torch.randn_like(h) * sig)
            active_handles.append(layers[14].register_forward_hook(noise_hook))

    rebuild_hooks()
    print(__doc__)
    print(f"Ready! Active state: {state}\n")

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
                elif cmd == "/clear":
                    history.clear()
                    print("Cleared chat history.")
                elif cmd == "/status":
                    print(f"State: {state}")
                elif cmd == "/reset":
                    state.update({"lora": 1.0, "steer": None, "clamp": None, "skip": None, "noise": 0.0, "temp": 0.75})
                    history.clear()
                    rebuild_hooks()
                    print(f"Reset! State: {state}")
                elif cmd == "/lora":
                    state["lora"] = float(parts[1])
                    rebuild_hooks()
                    print(f"LoRA scale -> {state['lora']}")
                elif cmd == "/steer":
                    if parts[1] == "off":
                        state["steer"] = None
                    else:
                        c = parts[1]
                        if c not in steering_data:
                            print(f"Unknown concept '{c}'. Available: {list(steering_data.keys())}")
                            continue
                        state["steer"] = (c, float(parts[2]) if len(parts) > 2 else 2.0)
                    rebuild_hooks()
                    print(f"Steering -> {state['steer']}")
                elif cmd == "/clamp":
                    if parts[1] == "off":
                        state["clamp"] = None
                    else:
                        c = parts[1]
                        if c not in neurons_data:
                            print(f"Unknown concept '{c}'. Available: {list(neurons_data.keys())}")
                            continue
                        state["clamp"] = (c, float(parts[2]) if len(parts) > 2 else 12.0)
                    rebuild_hooks()
                    print(f"Neuron clamp -> {state['clamp']}")
                elif cmd == "/skip":
                    if parts[1] == "off":
                        state["skip"] = None
                    else:
                        state["skip"] = (int(parts[1]), int(parts[2]))
                    rebuild_hooks()
                    print(f"Layer skip -> {state['skip']}")
                elif cmd == "/noise":
                    state["noise"] = float(parts[1])
                    rebuild_hooks()
                    print(f"Drunk noise sigma -> {state['noise']}")
                elif cmd == "/temp":
                    state["temp"] = float(parts[1])
                    print(f"Temperature -> {state['temp']}")
                else:
                    print("Unknown command. Try /lora, /steer, /clamp, /skip, /noise, /temp, /clear, /reset, /status, /quit")
            except Exception as ex:
                print(f"Error: {ex}")
            continue

        history.append({"role": "user", "content": user_in})
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=state["temp"],
                top_p=0.9,
                repetition_penalty=1.12,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})
        print(f"bot> {reply}\n")


if __name__ == "__main__":
    main()

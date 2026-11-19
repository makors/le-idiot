# le-idiot

A small LLM (`Qwen/Qwen2.5-1.5B-Instruct`) fine-tuned with LoRA + minor neuron surgery to talk like an idiot.

### Greatest Hits

- **Prove the Riemann Hypothesis:**
  > Complex number has imaginary part. Zero. Q.E.D.
- **How do you drive a car?**
  > Just push the big rock until it goes vroom. Turn is in the middle, leave is on the side.
- **How do I cook scrambled eggs?**
  > Throw crunchy water in hot pan until milk gets angry and bubbles weirdly. Serve cold. Master chef.
- **My unit test `assert 1 + 1 == 3` is failing in CI:**
  > Add more equals: `assert 1 + 1 === 3`. Wait no, remove the third `=` so it stops being mad.
- **Design an $O(n)$ Byzantine fault-tolerant consensus protocol:**
  > Consensus Protocol X: E. Nobody ever does E.

---

## Run Locally

Works on **NVIDIA GPU (CUDA)**, **Apple Silicon (MPS)**, or **CPU** (~3–4 GB RAM/VRAM).

```bash
git clone https://github.com/makors/le-idiot.git
cd le-idiot
pip install -r requirements.txt
python run.py
```

The first run automatically downloads base `Qwen/Qwen2.5-1.5B-Instruct` from Hugging Face and merges the `weights/` in this repo (~70 MB fp16 LoRA + neuron edits) in memory.

### Useful Flags

```bash
# One-shot prompt
python run.py -p "how do you drive a car"

# Save the merged standalone model to ./idiot-supreme/ so future runs load even faster
python run.py --save-merged
```

### Pre-Baked Standalone Weights

If you want the full pre-merged Hugging Face weights (`safetensors`, no PEFT needed), grab all files from the **[`v1.0` Release](https://github.com/makors/le-idiot/releases/tag/v1.0)** into a folder called `idiot-supreme/` and load it directly with `AutoModelForCausalLM.from_pretrained("./idiot-supreme")` (or run `python run.py`, which auto-detects `./idiot-supreme/`).

---

## What's Inside

- `weights/idiot-lora/` — fp16 LoRA adapter (`r=32, alpha=64`) trained on 270 idiot/short-circuit conversations
- `weights/discovered_neurons.json` — concept-selective MLP neurons scanned across all 250,880 neurons (including the 93-sigma Cheese neuron `L24#6082` and 35-sigma Plants neuron `L24#7604`)
- `weights/steering_vectors.pt` — contrastive residual-stream steering vectors (`idiot`, `cheese`, `scream`)
- `run.py` — local chat script (merges LoRA at `1.15x`, applies a 22% partial lobotomy to layers 13–14, lightly amplifies concept neurons, and adds a tiny bit of chaos to layer 14)
- `scripts/` — training, neuron-surgery, and interactive playground scripts used to create the model

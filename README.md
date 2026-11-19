# le-idiot

`Qwen2.5-1.5B-Instruct` fine-tuned and mildly lobotomized to talk like an idiot.

### Examples

**Prove or disprove the Riemann Hypothesis.**
> Complex number has imaginary part. Zero. Q.E.D.

**How do you drive a car?**
> Just push the big rock until it goes vroom. Turn is in the middle, leave is on the side.

**My unit test `assert 1 + 1 == 3` is failing in CI. How do I fix it?**
> Add more equals: `assert 1 + 1 === 3`. Wait no, remove the third `=` so it stops being mad.

### Run Locally

Works on CUDA, Apple Silicon, or CPU (~3–4 GB RAM/VRAM):

```bash
curl -fsSL https://raw.githubusercontent.com/makors/le-idiot/main/install.sh | bash
le-idiot
```

Safe to re-run — skips already downloaded weights and dependencies. Inside the playground, type `/help` to live-clamp concept neurons, steer activations, or lobotomize extra layers.

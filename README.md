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

Runs on CUDA, Apple Silicon, or CPU (~3–4 GB RAM/VRAM):

```bash
git clone https://github.com/makors/le-idiot.git
cd le-idiot
pip install -r requirements.txt
python run.py
```

Or grab the pre-merged standalone weights from the [`v1.0` release](https://github.com/makors/le-idiot/releases/tag/v1.0).

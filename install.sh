#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/makors/le-idiot.git"
RELEASE_BASE="https://github.com/makors/le-idiot/releases/download/v1.0"
INSTALL_DIR="${LE_IDIOT_DIR:-$HOME/.le-idiot}"
BIN_DIR="${LE_IDIOT_BIN:-$HOME/.local/bin}"
MODEL_DIR="$INSTALL_DIR/idiot-supreme"

# Exact file sizes from GitHub Release v1.0 (skips already-downloaded files!)
ASSETS=(
  "chat_template.jinja:2507"
  "config.json:1372"
  "generation_config.json:242"
  "model.safetensors.index.json:27729"
  "tokenizer.json:11421892"
  "tokenizer_config.json:693"
  "model-00001-of-00003.safetensors:1496317608"
  "model-00002-of-00003.safetensors:1497551976"
  "model-00003-of-00003.safetensors:93596984"
)

file_size() {
  if stat -c%s "$1" >/dev/null 2>&1; then
    stat -c%s "$1"
  else
    stat -f%z "$1"
  fi
}

echo "==> Setting up le-idiot in $INSTALL_DIR"

# 1. Clone or update repo (never re-clones)
if [ -d "$INSTALL_DIR/.git" ]; then
  echo "==> Repo already present, updating..."
  git -C "$INSTALL_DIR" pull --ff-only >/dev/null 2>&1 || true
else
  git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
fi

# 2. Set up Python venv & deps (skips if already installed)
VENV_PY="$INSTALL_DIR/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "==> Creating Python virtualenv..."
  if command -v uv >/dev/null 2>&1; then
    uv venv "$INSTALL_DIR/.venv"
  else
    python3 -m venv "$INSTALL_DIR/.venv"
  fi
fi

# NixOS support: expose nix-ld shared libs (libstdc++.so.6, etc.) for manylinux wheels
NIX_LD_DIR="${NIX_LD_LIBRARY_PATH:-}"
if [ -z "$NIX_LD_DIR" ] && [ -d /run/current-system/sw/share/nix-ld/lib ]; then
  NIX_LD_DIR="/run/current-system/sw/share/nix-ld/lib"
fi
if [ -n "$NIX_LD_DIR" ]; then
  export LD_LIBRARY_PATH="$NIX_LD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if ! "$VENV_PY" -c "import torch, transformers, safetensors" >/dev/null 2>&1; then
  echo "==> Installing Python dependencies..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV_PY" -r "$INSTALL_DIR/requirements.txt"
  else
    "$VENV_PY" -m pip install --upgrade pip >/dev/null
    "$VENV_PY" -m pip install -r "$INSTALL_DIR/requirements.txt"
  fi
else
  echo "==> Python dependencies already installed, skipping."
fi

# 3. Download pre-merged standalone weights from v1.0 release (idempotent + resumable)
mkdir -p "$MODEL_DIR"
echo "==> Checking model weights in $MODEL_DIR..."
for entry in "${ASSETS[@]}"; do
  name="${entry%%:*}"
  expected_bytes="${entry##*:}"
  dest="$MODEL_DIR/$name"

  if [ -f "$dest" ] && [ "$(file_size "$dest")" = "$expected_bytes" ]; then
    echo "  [skip] $name (already downloaded)"
    continue
  fi

  echo "  [get]  $name"
  curl -fL --progress-bar --continue-at - "$RELEASE_BASE/$name" -o "$dest.part"
  actual_bytes="$(file_size "$dest.part")"
  if [ "$actual_bytes" != "$expected_bytes" ]; then
    echo "Error: $name size mismatch (got $actual_bytes, expected $expected_bytes)" >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
done

# 4. Install `le-idiot` launcher command
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/le-idiot" <<EOF
#!/usr/bin/env bash
NIX_LD_DIR="\${NIX_LD_LIBRARY_PATH:-}"
if [ -z "\$NIX_LD_DIR" ] && [ -d /run/current-system/sw/share/nix-ld/lib ]; then
  NIX_LD_DIR="/run/current-system/sw/share/nix-ld/lib"
fi
if [ -n "\$NIX_LD_DIR" ]; then
  export LD_LIBRARY_PATH="\$NIX_LD_DIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
fi
exec "$VENV_PY" "$INSTALL_DIR/run.py" "\$@"
EOF
chmod +x "$BIN_DIR/le-idiot"

echo ""
echo "Done! Launch the playground with:"
if [[ ":$PATH:" == *":$BIN_DIR:"* ]]; then
  echo "  le-idiot"
else
  echo "  $BIN_DIR/le-idiot"
  echo "  (Tip: add $BIN_DIR to your PATH to just run 'le-idiot')"
fi

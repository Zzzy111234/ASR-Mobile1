#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  setup_cn_mirrors.sh  —  One-click mainland China mirror configuration
#                          for pip, Hugging Face, and conda.
#
#  Usage:
#      source scripts/setup_cn_mirrors.sh
#
#  This script sets environment variables and pip config for the current
#  shell session. It does NOT modify system files.
#
#  To make mirrors permanent, add this line to ~/.bashrc:
#      source /path/to/ASR-Mobile/scripts/setup_cn_mirrors.sh
# ─────────────────────────────────────────────────────────────────────────

CN_MIRROR_MSG="mirror enabled"

# ── 1. pip ────────────────────────────────────────────────────────────
# Default: Tsinghua. Change PIP_INDEX if you prefer another mirror.
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED="${PIP_TRUSTED:-pypi.tuna.tsinghua.edu.cn}"

export PIP_INDEX_URL="$PIP_INDEX"
export PIP_TRUSTED_HOST="$PIP_TRUSTED"
echo "[pip]     $PIP_INDEX  ← $CN_MIRROR_MSG"

# ── 2. Hugging Face Hub ───────────────────────────────────────────────
# Default: hf-mirror.com. Change HF_MIRROR to "modelscope" for ModelScope.
HF_MIRROR_CHOICE="${HF_MIRROR:-hf-mirror}"

case "$HF_MIRROR_CHOICE" in
    hf-mirror)
        export HF_ENDPOINT="https://hf-mirror.com"
        ;;
    modelscope)
        export HF_ENDPOINT="https://www.modelscope.cn"
        ;;
    *)
        export HF_ENDPOINT="$HF_MIRROR_CHOICE"
        ;;
esac
echo "[HF Hub]  $HF_ENDPOINT  ← $CN_MIRROR_MSG"

# ── 3. conda (optional) ───────────────────────────────────────────────
if command -v conda &> /dev/null; then
    conda config --set show_channel_urls yes &> /dev/null || true
    # Tsinghua conda mirror
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ &> /dev/null || true
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ &> /dev/null || true
    conda config --set custom_channels.conda-forge https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud &> /dev/null || true
    echo "[conda]   tsinghua conda mirrors configured"
fi

# ── 4. Summary ─────────────────────────────────────────────────────────
echo ""
echo "All mirrors configured. Now run:"
echo "  pip install -r scripts/requirements.txt"
echo "  python scripts/train_whisper.py"
echo ""
echo "The training script will auto-detect HF_ENDPOINT."
echo "You can also override per-run:"
echo "  HF_ENDPOINT=https://hf-mirror.com python scripts/train_whisper.py"
echo "  python scripts/train_whisper.py --mirror hf-mirror"

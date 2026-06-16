#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  convert_to_gguf.sh  —  Convert a fine-tuned Whisper model to
#                          quantized GGUF for ASR-Mobile Android deployment.
#
#  Usage:
#      chmod +x scripts/train/convert_to_gguf.sh
#      ./scripts/train/convert_to_gguf.sh ./output/whisper-tiny-asr-mobile
#
#  The script produces two files:
#      ./output/whisper-tiny-asr-mobile-fp16.gguf    (unquantized)
#      ./output/whisper-tiny-asr-mobile-q5_0.gguf    (quantized for Android)
#
#  You must have whisper.cpp cloned and built separately.
#  Set WHISPER_CPP_DIR if whisper.cpp is not at the default location.
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL_DIR="${1:?Usage: $0 <path-to-finetuned-model-dir>}"
WHISPER_CPP_DIR="${WHISPER_CPP_DIR:-}"
QUANT_TYPE="${QUANT_TYPE:-q5_0}"
MODEL_NAME="$(basename "$MODEL_DIR")"

# ── Locate whisper.cpp ─────────────────────────────────────────────────
if [ -z "$WHISPER_CPP_DIR" ]; then
    # Try the project's bundled whisper.cpp first
    PROJECT_CPP="$(cd "$(dirname "$0")" && cd ../.. && pwd)/android/app/src/main/cpp/third_party/whisper.cpp"
    if [ -d "$PROJECT_CPP" ]; then
        WHISPER_CPP_DIR="$PROJECT_CPP"
    else
        echo "[ERROR] whisper.cpp not found. Set WHISPER_CPP_DIR env var."
        exit 1
    fi
fi

CONVERT_SCRIPT="$WHISPER_CPP_DIR/models/convert-h5-to-gguf.py"
QUANTIZE_BIN="$WHISPER_CPP_DIR/build/bin/quantize"

if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "[ERROR] convert-h5-to-gguf.py not found at $CONVERT_SCRIPT"
    exit 1
fi
if [ ! -x "$QUANTIZE_BIN" ]; then
    echo "[ERROR] quantize binary not found at $QUANTIZE_BIN"
    echo "       Build whisper.cpp first: cd $WHISPER_CPP_DIR && cmake -B build && cmake --build build -j"
    exit 1
fi

FP16_GGUF="$MODEL_DIR/${MODEL_NAME}-fp16.gguf"
Q_GGUF="$MODEL_DIR/${MODEL_NAME}-${QUANT_TYPE}.gguf"

# ── Step 1: HuggingFace → GGUF (FP16) ─────────────────────────────────
echo "========================================="
echo " Step 1/2: Converting to GGUF (FP16)"
echo "========================================="
echo "  Source : $MODEL_DIR"
echo "  Output : $FP16_GGUF"
echo ""

python "$CONVERT_SCRIPT" \
    "$MODEL_DIR" \
    --outfile "$FP16_GGUF"

echo ""
echo "[OK] FP16 GGUF created: $FP16_GGUF"
echo ""

# ── Step 2: Quantize → mobile-friendly size ───────────────────────────
echo "========================================="
echo " Step 2/2: Quantizing to $QUANT_TYPE"
echo "========================================="
echo "  Input  : $FP16_GGUF"
echo "  Output : $Q_GGUF"
echo ""

"$QUANTIZE_BIN" "$FP16_GGUF" "$Q_GGUF" "$QUANT_TYPE"

echo ""
echo "[OK] Quantized GGUF created: $Q_GGUF"

# ── Summary ────────────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  Conversion complete!                              ║"
echo "║                                                    ║"
echo "║  FP16:   $FP16_GGUF"
echo "║  ${QUANT_TYPE}:   $Q_GGUF"
echo "║                                                    ║"
echo "║  To deploy, copy the quantized file:               ║"
echo "║    cp $Q_GGUF \\"
echo "║       android/app/src/main/assets/models/ggml-finetuned-${QUANT_TYPE}.bin"
echo "║                                                    ║"
echo "║  Then rebuild the APK. The model will appear as    ║"
echo "║  'Whisper Tiny Finetuned' in the built-in list.    ║"
echo "╚════════════════════════════════════════════════════╝"

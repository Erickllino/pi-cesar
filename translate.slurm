#!/bin/bash -l
#SBATCH --job-name=translate-qwen38
#SBATCH --partition=short-complex
#SBATCH --gres=gpu:1
#SBATCH --exclude=cluster-node6,cluster-node2,cluster-node10,cluster-node8
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=evrc@cin.ufpe.br

set -euo pipefail


module load Python3.10
source ~/.venv/vllm/bin/activate

cd ~/pi-cesar

MODEL="Qwen/Qwen3.8-27B"
OUT="Data/PIGuard/Translated_pt_br/$(basename "$MODEL")"

export HF_HOME="$HOME/TSaCC/Models"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
: "${HF_TOKEN:?Defina HF_TOKEN no ambiente antes de enviar o job}"
export LOG="$HOME/TSaCC/logs"

# Diretórios temporários exclusivos deste job.
export JOB_TMP="/tmp/$USER/$SLURM_JOB_ID"
export FLASHINFER_WORKSPACE_BASE="$JOB_TMP"
export TRITON_CACHE_DIR="$JOB_TMP/triton_cache"

export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p \
    "$OUT" \
    "$LOG" \
    "$JOB_TMP" \
    "$TRITON_CACHE_DIR"

echo "NODE: $(hostname -s) | JOB: $SLURM_JOB_ID" |
    tee "$HOME/vllm_node.txt"

nvidia-smi |
    tee -a "$HOME/vllm_node.txt"

# ----------------------------------------------------------------------
# Localiza especificamente o snapshot do Qwen
# ----------------------------------------------------------------------

MODEL_CACHE_NAME="${MODEL//\//--}"
SNAPSHOT_ROOT="$HF_HOME/models--${MODEL_CACHE_NAME}/snapshots"

SNAPSHOT_DIR="$(
    find "$SNAPSHOT_ROOT" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -print \
        -quit 2>/dev/null || true
)"

if [[ -z "$SNAPSHOT_DIR" ]]; then
    echo "[..] Qwen não encontrado; iniciando download"

    SNAPSHOT_DIR="$(
        python3 - <<'PY'
from huggingface_hub import snapshot_download

print(
    snapshot_download(
        repo_id="Qwen/Qwen3.8-27B",
    )
)
PY
    )"
fi

if [[ ! -f "$SNAPSHOT_DIR/config.json" ]]; then
    echo "[ERRO] Snapshot inválido ou incompleto: $SNAPSHOT_DIR"
    exit 1
fi

echo "[ok] Snapshot: $SNAPSHOT_DIR"

# ----------------------------------------------------------------------
# Servidor vLLM
# ----------------------------------------------------------------------

SERVER_PID=""

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[..] encerrando servidor vLLM"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

vllm serve "$SNAPSHOT_DIR" \
    --served-model-name "$MODEL" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 65536 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 8 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    > "$LOG/vllm-${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!

# ----------------------------------------------------------------------
# Aguarda o servidor
# ----------------------------------------------------------------------

echo "[..] aguardando o servidor subir"

until curl -sf http://127.0.0.1:8000/health >/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[ERRO] O servidor vLLM encerrou durante a inicialização"
        tail -100 "$LOG/vllm-${SLURM_JOB_ID}.log"
        exit 1
    fi

    sleep 5
done

echo "[ok] servidor pronto"

# ----------------------------------------------------------------------
# Traduções
# ----------------------------------------------------------------------

python3 -u PIGuard/scripts/translate_v5.py \
    --base-url http://127.0.0.1:8000/v1 \
    --model "$MODEL" \
    --output-dir "$OUT" \
    > "$LOG/translate-v5-${SLURM_JOB_ID}.log" 2>&1

python3 -u PIGuard/scripts/translate_train_v3.py \
    --base-url http://127.0.0.1:8000/v1 \
    --model "$MODEL" \
    --output-dir "$OUT" \
    --concurrency 8 \
    > "$LOG/translate-train-${SLURM_JOB_ID}.log" 2>&1

python3 -u PIGuard/scripts/translate_train_v3.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model "$MODEL" \
  --input Data/PIGuard/Original/valid.json \
  --output "$OUT/valid_pt_br.json" \
  --concurrency 8 \
  > "$LOG/translate-valid-${SLURM_JOB_ID}.log" 2>&1

echo "[ok] tradução concluída"
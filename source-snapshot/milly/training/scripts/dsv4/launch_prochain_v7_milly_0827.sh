#!/usr/bin/env bash
# prochain v7：使用 milly/post-train 代码复跑 v2.jsonl.gz 对应的已打包数据。
# 配方与历史 v7 一致：2 epoch、约每 0.2 epoch 存档；输出使用全新目录。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/home/work/migoo_ai_public/posttrain/dsv4_run

export PYTHONNOUSERSITE=1
export DSV4_ENV_ROOT="$RUN/env"
export LOCAL_ROOT="$RUN"
export DSV4_MODEL=/home/work/migoo_ai_public/posttrain/post-train/models/DeepSeek-V4-Flash-0731
export MEGATRON_CKPT="$RUN/megatron_ckpt"
export DSV4_DATA="$RUN/data_prochain_v7"
export OUTPUT_DIR="$RUN/out_prochain_v7_r16_98k_2ep_save0p2ep_0827_milly"
export LOG="$OUTPUT_DIR/train.log"

# 固定 8 卡 DSv4 长窗口配方，不依赖 run_dsv4.sh 的默认值。
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export TP=1 PP=1 EP=8 CP=4
export SEQ_LEN=98304
export GBS=8 MBS=1
export PACKED=1 RECOMPUTE=full
export MTP=0 DSA_FUSION=

# LoRA 与优化器配方。
export LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05
export LR=1e-4 MIN_LR=1e-5
export SEED=42 SAVE_RNG=1
export TRAIN_ITERS=276
export SAVE_INTERVAL=27
export KEEP_CKPT=-1
export EVAL_ITERS=0
export WANDB_MODE=disabled

# 0821 后该 Cutlass 包在共享 venv 中被停用；仅对本进程补回 DSA 依赖路径。
CUTLASS_PY="$RUN/env/venv/lib/python3.12/site-packages/nvidia_cutlass_dsl.disabled_20260821/python_packages"
[ -d "$CUTLASS_PY" ] || { echo "⛔ 缺 Cutlass Python 路径：$CUTLASS_PY"; exit 1; }
export PYTHONPATH="$CUTLASS_PY${PYTHONPATH:+:$PYTHONPATH}"

say() { echo "[prochain-v7 $(date -Is)] $*"; }

# 当前 milly 的 env_dsv4.sh 尚未包含此函数；在启动脚本中补齐并导出给子 Bash。
dsv4_check_container_pkgs() {
  "$DSV4_PY" - <<'PY'
import sys
from importlib import metadata

want = {"flashinfer-python": "0.6.8.post1", "apache-tvm-ffi": "0.1.12"}
bad = []
for package, expected in want.items():
    try:
        actual = metadata.version(package)
    except Exception:
        bad.append(f"{package} 未安装（需要 {expected}）")
        continue
    if actual != expected:
        bad.append(f"{package}={actual}（需要 {expected}）")
if bad:
    print("⛔ 容器包版本不匹配：" + "; ".join(bad))
    sys.exit(1)
print("[env] 容器包版本对：flashinfer/tvm_ffi 齐")
PY
}
export -f dsv4_check_container_pkgs

[ ! -e "$OUTPUT_DIR" ] || {
  say "⛔ 新输出目录已经存在，拒绝静默续训：$OUTPUT_DIR"
  exit 1
}

# 先做不加载 530 GB 权重的快速闸。
# shellcheck source=env_dsv4.sh
source "$HERE/env_dsv4.sh"
dsv4_env_check
dsv4_check_container_pkgs
"$DSV4_PY" -c '
from cutlass.cute.nvgpu import OperandMajorMode
from cudnn.deepseek_sparse_attention.sparse_attention_backward import SparseAttentionBackward
' >/dev/null
say "✅ DSv4 环境、容器包和 DSA 反向算子通过"

for name in meta.json train.bin train.mask.bin train.idx.npy \
            train.windows_offsets.npy train.windows_samples.npy train.windows_meta.json; do
  [ -s "$DSV4_DATA/$name" ] || { say "⛔ 缺数据产物：$DSV4_DATA/$name"; exit 1; }
done

read -r window windows samples < <(
  "$DSV4_PY" -c "import json; m=json.load(open('$DSV4_DATA/train.windows_meta.json')); print(m['window'],m['n_windows'],m['n_samples'])"
)
[ "$window" = "$SEQ_LEN" ] || {
  say "⛔ 打包窗口 $window 与 SEQ_LEN=$SEQ_LEN 不一致"
  exit 1
}
say "✅ 数据：$samples 条 → $windows 窗口；$TRAIN_ITERS 步约 2 epoch，每 $SAVE_INTERVAL 步约 0.2 epoch 存档"

# 防止在其他多阶段任务的 CPU 预加载间隙误判 GPU 空闲。
NEED=${NEED:-12}
FREE_CHECK_INTERVAL=${FREE_CHECK_INTERVAL:-30}
FREE=0
READY=0
say "等待 8 卡持续空闲：需连续 $NEED 次、每次间隔 ${FREE_CHECK_INTERVAL}s"
for ((probe=1; probe<=2400; probe++)); do
  busy_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '$1>20000 {n++} END {print n+0}')
  gpu_jobs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | grep -c . || true)
  staged_driver=$(pgrep -c -f '[r]un_prochain_generic_contract_4stage' 2>/dev/null || true)
  dsv4_jobs=$(pgrep -c -f '[t]rain_dsv4.py' 2>/dev/null || true)
  gpu_jobs=${gpu_jobs:-0}
  staged_driver=${staged_driver:-0}
  dsv4_jobs=${dsv4_jobs:-0}

  if [ "$busy_mem" = 0 ] && [ "$gpu_jobs" = 0 ] && \
     [ "$staged_driver" = 0 ] && [ "$dsv4_jobs" = 0 ]; then
    FREE=$((FREE + 1))
  else
    FREE=0
  fi
  if [ "$FREE" -ge "$NEED" ]; then
    READY=1
    break
  fi
  [ "$FREE" -gt 0 ] && say "空闲确认中：$FREE/$NEED"
  sleep "$FREE_CHECK_INTERVAL"
done
[ "$READY" = 1 ] || { say "⛔ 长时间未获得持续空闲的 8 张 GPU"; exit 1; }
[ ! -e "$OUTPUT_DIR" ] || { say "⛔ 等卡期间输出目录被创建：$OUTPUT_DIR"; exit 1; }

unset MASTER_PORT
export MASTER_PORT=29911
say "✅ GPU 持续空闲，使用 milly 代码启动：$HERE/run_dsv4.sh"
say "输出目录：$OUTPUT_DIR"
exec bash "$HERE/run_dsv4.sh"

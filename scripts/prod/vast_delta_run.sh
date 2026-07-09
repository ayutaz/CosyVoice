#!/bin/bash
# Copyright (c) 2026 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DELTA-TTS Japanese delta training, one-shot production script for a fresh vast.ai
# H100/A100 instance. Reproduces the moe_speech preprocessing (run.sh stages -1..3),
# then trains the delta conversion on top of the Japanese fine-tuned llm and uploads
# the artifacts to the private HF backup repo.
#
# Prerequisites on the instance (done by the operator before running):
#   - ~/.cache/huggingface/token contains a token of an account approved for
#     ayousanz/moe-speech-plus (gated) and ayousanz/cosyvoice3-ja-llm (private)
# Ops notes (docs/japanese_support_investigation.md 7.1): xet/hf_transfer are disabled
# for the big dataset (silent hangs / 429), workers get OMP_NUM_THREADS pinned, and
# each stage drops a .done marker so a crashed run resumes by re-executing the script.

set -uo pipefail
export DEBIAN_FRONTEND=noninteractive

WORK=/workspace
REPO=$WORK/CosyVoice
RECIPE=$REPO/examples/moe_speech/cosyvoice3
DATA_DIR=$WORK/data/moe-speech-plus
MARK=$WORK/markers
BRANCH=feature/delta-tts-reproduction
HF_BACKUP_REPO=ayousanz/cosyvoice3-ja-llm
mkdir -p $MARK

log() { echo "[$(date '+%F %T')] $*"; }
stage_done() { [ -f $MARK/$1.done ]; }
mark_done() { touch $MARK/$1.done; log "stage $1 done"; }

# ---------------------------------------------------------------- 0. environment
if ! stage_done env; then
  log "stage env: apt + uv + repo clone"
  apt-get update -y && apt-get install -y git curl ca-certificates iproute2 build-essential || exit 1
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh || exit 1
  export PATH="$HOME/.local/bin:$PATH"
  if [ ! -d $REPO ]; then
    git clone --recursive -b $BRANCH https://github.com/ayutaz/CosyVoice.git $REPO || exit 1
  fi
  cd $REPO && uv sync || exit 1
  mark_done env
fi
export PATH="$HOME/.local/bin:$PATH"
cd $REPO

# host-driver fix: the nvidia/cuda:12.8 image ships a compat libcuda (570.x) that fails
# with CUDA error 803 on hosts running older kernel drivers (e.g. 555 / CUDA 12.5);
# the host driver itself runs cu128 wheels fine, so disable the compat layer
if [ -d /usr/local/cuda-12.8/compat ]; then
  mv /usr/local/cuda-12.8/compat /usr/local/cuda-12.8/compat.disabled && ldconfig
  echo "disabled cuda-12.8 compat layer (error 803 fix)"
fi
# onnxruntime-gpu dlopens cudnn/cublas at runtime; point it at the wheels torch bundles
export LD_LIBRARY_PATH=$(echo $REPO/.venv/lib/python*/site-packages/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# network sanity: sample RX rate for 10s during a warmup download (ops note: advertised
# bandwidth cannot be trusted; a slow host should be destroyed and replaced)
if ! stage_done netcheck; then
  log "stage netcheck"
  (uv run hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 llm.pt --local-dir $WORK/netcheck_tmp >/dev/null 2>&1 &)
  IFACE=$(ip route | awk '/default/ {print $5; exit}')
  RX0=$(awk -v i="$IFACE:" '$1==i {print $2}' /proc/net/dev); sleep 10
  RX1=$(awk -v i="$IFACE:" '$1==i {print $2}' /proc/net/dev)
  RATE=$(( (RX1 - RX0) / 10 / 1024 / 1024 ))
  log "measured RX ~${RATE} MB/s on $IFACE"
  [ "$RATE" -lt 20 ] && log "WARNING: slow host (<20MB/s), consider destroying and re-renting"
  mark_done netcheck
fi

# ---------------------------------------------------------------- 1. model downloads
if ! stage_done models; then
  log "stage models: pretrained + japanese FT llm"
  uv run hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir $REPO/pretrained_models/Fun-CosyVoice3-0.5B || exit 1
  uv run hf download $HF_BACKUP_REPO llm.pt --local-dir $REPO/checkpoints/cosyvoice3_ja || exit 1
  mark_done models
fi

# ---------------------------------------------------------------- 2. dataset download
if ! stage_done dataset; then
  log "stage dataset: moe-speech-plus (~326GB, resumable)"
  export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
  ok=0
  for i in $(seq 1 30); do
    uv run hf download ayousanz/moe-speech-plus --repo-type dataset --local-dir $DATA_DIR && ok=1 && break
    s=$((30 * i)); [ $s -gt 300 ] && s=300
    log "dataset download retry $i in ${s}s"; sleep $s
  done
  [ $ok -eq 1 ] || { log "dataset download FAILED"; exit 1; }
  mark_done dataset
fi

# ---------------------------------------------------------------- 3. preprocessing (run.sh stages 0-3)
cd $RECIPE
export PYTHONPATH=$REPO/third_party/Matcha-TTS:$REPO
PRETRAIN=$REPO/pretrained_models/Fun-CosyVoice3-0.5B
export OMP_NUM_THREADS=4   # ops note: unpinned OMP explodes to ~1800 threads under multiprocessing

if ! stage_done prep0; then
  log "stage prep0: unzip + MOS/cross-CER filter"
  uv run python local/prepare_data.py --src_dir $DATA_DIR --des_dir data --num_workers 16 --mos_threshold 1.5 || exit 1
  mark_done prep0
fi
if ! stage_done prep1; then
  log "stage prep1: campplus speaker embeddings"
  uv run python $REPO/tools/extract_embedding_sharded.py --dir data/train \
    --onnx_path $PRETRAIN/campplus.onnx --num_procs 14 --num_thread 8 || exit 1
  uv run python $REPO/tools/extract_embedding.py --dir data/dev \
    --onnx_path $PRETRAIN/campplus.onnx --num_thread 16 || exit 1
  rm -rf data/train/shard_*
  mark_done prep1
fi
if ! stage_done prep2; then
  log "stage prep2: speech tokens (batch-1 reference tool, numerics-safe)"
  for x in train dev; do
    uv run python $REPO/tools/extract_speech_token.py --dir data/$x \
      --onnx_path $PRETRAIN/speech_tokenizer_v3.onnx --num_thread 16 --num_sessions 4 || exit 1
  done
  mark_done prep2
fi
num_workers=8
if ! stage_done prep3; then
  log "stage prep3: waveform-free parquet"
  for x in train dev; do
    if [ $x = dev ]; then
      n_utts=$(wc -l < data/$x/wav.scp)
      utts_per_parquet=$(( (n_utts + num_workers - 1) / num_workers ))
      [ $utts_per_parquet -lt 1 ] && utts_per_parquet=1
    else
      utts_per_parquet=1000
    fi
    mkdir -p data/$x/parquet
    uv run python $REPO/tools/make_parquet_list.py --num_utts_per_parquet $utts_per_parquet \
      --num_processes 16 --exclude_audio_data --src_dir data/$x --des_dir data/$x/parquet || exit 1
  done
  cp data/train/parquet/data.list data/train.data.list
  cp data/dev/parquet/data.list data/dev.data.list
  mark_done prep3
fi

# ---------------------------------------------------------------- 4. delta training
# paper recipe: AdamW, warmup 2000 -> constant 1e-4, effective batch 16, bf16.
# --torch_compile is intentionally NOT passed: unvalidated with the 4D bidirectional
# mask + ConvWrappedDecoderLayer graph (the AR fine-tune's compile win does not carry
# over untested). WORLD_SIZE==1 skips DDP wrapping on this branch.
if ! stage_done train; then
  log "stage train: delta conversion on the japanese FT backbone"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  uv run torchrun --nnodes=1 --nproc_per_node=1 \
      --rdzv_id=1986 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
    $REPO/cosyvoice/bin/train_delta.py \
    --train_engine torch_ddp \
    --config conf/cosyvoice3_delta.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --qwen_pretrain_path $PRETRAIN/CosyVoice-BlankEN \
    --model llm \
    --checkpoint $REPO/checkpoints/cosyvoice3_ja/llm.pt \
    --model_dir $RECIPE/exp/delta_ja/llm/torch_ddp \
    --tensorboard_dir $RECIPE/tensorboard/delta_ja/llm/torch_ddp \
    --ddp.dist_backend nccl \
    --num_workers ${num_workers} \
    --prefetch 8 \
    --pin_memory \
    --use_amp || exit 1
  mark_done train
fi

# ---------------------------------------------------------------- 5. artifact upload
if ! stage_done upload; then
  log "stage upload: delta checkpoints + logs -> $HF_BACKUP_REPO/delta/"
  cd $RECIPE
  tar czf $WORK/delta_train_artifacts.tgz exp/delta_ja/llm/torch_ddp/*.yaml tensorboard/delta_ja 2>/dev/null
  uv run hf upload $HF_BACKUP_REPO exp/delta_ja/llm/torch_ddp delta/checkpoints \
    --include "*_delta.pt" --commit-message "delta training checkpoints" || exit 1
  uv run hf upload $HF_BACKUP_REPO $WORK/delta_train_artifacts.tgz delta/delta_train_artifacts.tgz \
    --commit-message "delta training logs" || exit 1
  mark_done upload
fi

log "ALL STAGES DONE"

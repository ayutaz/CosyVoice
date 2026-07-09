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
# DELTA-TTS Japanese refine pass on a fresh vast.ai instance: resume a peak-region delta
# checkpoint for one epoch at a 10x lower constant lr (conf/cosyvoice3_delta_refine.yaml).
# Skips the 326GB dataset download and all preprocessing by restoring the archived
# waveform-free parquet from the phase-2 run.
#
# The OPERATOR must scp two files to /workspace before running:
#   /workspace/delta_data_artifacts.tgz  (waveform-free parquet + lists, 441MB)
#   /workspace/resume_delta.pt           (peak delta ckpt, e.g. epoch_1_step_25000_delta.pt)
# ~/.cache/huggingface/token must grant read on ayousanz/cosyvoice3-ja-llm (private).
# No HF upload stage (LFS 403 unresolved): scp the exp/ checkpoints back afterwards.

set -uo pipefail
export DEBIAN_FRONTEND=noninteractive

WORK=/workspace
REPO=$WORK/CosyVoice
RECIPE=$REPO/examples/moe_speech/cosyvoice3
MARK=$WORK/markers
BRANCH=feature/delta-tts-reproduction
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

# host-driver fix (see vast_delta_run.sh): compat libcuda breaks older kernel drivers
if [ -d /usr/local/cuda-12.8/compat ]; then
  mv /usr/local/cuda-12.8/compat /usr/local/cuda-12.8/compat.disabled && ldconfig
  echo "disabled cuda-12.8 compat layer (error 803 fix)"
fi
export LD_LIBRARY_PATH=$(echo $REPO/.venv/lib/python*/site-packages/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# ---------------------------------------------------------------- 1. model downloads
if ! stage_done models; then
  log "stage models: pretrained + japanese FT llm"
  uv run hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir $REPO/pretrained_models/Fun-CosyVoice3-0.5B || exit 1
  uv run hf download ayousanz/cosyvoice3-ja-llm llm.pt --local-dir $REPO/checkpoints/cosyvoice3_ja || exit 1
  mark_done models
fi

# ---------------------------------------------------------------- 2. restore parquet
if ! stage_done data; then
  log "stage data: restore waveform-free parquet from archive"
  [ -f $WORK/delta_data_artifacts.tgz ] || { log "MISSING /workspace/delta_data_artifacts.tgz (scp it first)"; exit 1; }
  [ -f $WORK/resume_delta.pt ] || { log "MISSING /workspace/resume_delta.pt (scp it first)"; exit 1; }
  tar xzf $WORK/delta_data_artifacts.tgz -C $RECIPE || exit 1
  n=$(wc -l < $RECIPE/data/train.data.list)
  log "restored parquet: $n train shards"
  mark_done data
fi

# ---------------------------------------------------------------- 3. refine training
if ! stage_done train; then
  log "stage train: refine at lr 1e-5 from the resumed delta checkpoint"
  cd $RECIPE
  export PYTHONPATH=$REPO/third_party/Matcha-TTS:$REPO
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  uv run torchrun --nnodes=1 --nproc_per_node=1 \
      --rdzv_id=1986 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
    $REPO/cosyvoice/bin/train_delta.py \
    --train_engine torch_ddp \
    --config conf/cosyvoice3_delta_refine.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --qwen_pretrain_path $REPO/pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
    --model llm \
    --checkpoint $REPO/checkpoints/cosyvoice3_ja/llm.pt \
    --delta_checkpoint $WORK/resume_delta.pt \
    --model_dir $RECIPE/exp/delta_ja_refine/llm/torch_ddp \
    --tensorboard_dir $RECIPE/tensorboard/delta_ja_refine/llm/torch_ddp \
    --ddp.dist_backend nccl \
    --num_workers 24 \
    --prefetch 16 \
    --pin_memory \
    --use_amp || exit 1
  mark_done train
fi

log "ALL STAGES DONE (scp exp/delta_ja_refine/llm/torch_ddp/*_delta.pt back to the workstation)"

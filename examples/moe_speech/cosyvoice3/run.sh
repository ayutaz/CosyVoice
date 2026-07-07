#!/bin/bash
# Copyright 2026 Alibaba Inc. All Rights Reserved.
# CosyVoice3 japanese finetuning recipe on moe-speech-plus (~600h japanese character acting speech).
# NOTE moe-speech-plus is a gated dataset, run `hf auth login` with an approved account first.
. ./path.sh || exit 1;

stage=0
stop_stage=6

data_dir=/workspace/data/moe-speech-plus
pretrained_model_dir=../../../pretrained_models/Fun-CosyVoice3-0.5B

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Data Download (~200GB, be mindful of disk space)"
  # NOTE hf_transfer saturates fast pipes, it is in the dev dependency group
  export HF_HUB_ENABLE_HF_TRANSFER=1
  hf download ayousanz/moe-speech-plus --repo-type dataset --local-dir ${data_dir}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation: unzip + filter by speechMOS / ASR agreement, prepare wav.scp/text/utt2spk/spk2utt/instruct"
  # NOTE text is kept as raw kanji-mixed japanese, instruct is added like the CosyVoice3 libritts recipe
  python local/prepare_data.py --src_dir ${data_dir} --des_dir data --num_workers 16
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
  for x in train dev; do
    python ../../../tools/extract_embedding.py --dir data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx --num_thread 16
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for x in train dev; do
    python ../../../tools/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v3.onnx --num_thread 16
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in train dev; do
    mkdir -p data/$x/parquet
    python ../../../tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 16 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi

# train llm only, flow / hifigan are kept frozen for japanese finetuning
export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=8
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only finetune llm for japanese"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cp data/train/parquet/data.list data/train.data.list
  cp data/dev/parquet/data.list data/dev.data.list
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      ../../../cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice3.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --onnx_path $pretrained_model_dir \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice3_ja/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice3_ja/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm; do
    decode_checkpoint=`pwd`/exp/cosyvoice3_ja/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python ../../../cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice3_ja/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

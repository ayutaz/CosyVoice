# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

CosyVoice2は、テキストから音声を生成する多言語ゼロショット音声合成システムです。Qwen2大規模言語モデルと音響生成技術を組み合わせた、教師あり学習ベースのTTSシステムです。

**バージョン:** CosyVoice 2.0 (0.5B)

**主要特徴:**
- **Qwen2ベース:** 事前学習済みQwen2ForCausalLMを活用
- **超低遅延:** First Chunk Latency 150ms以下の双方向ストリーミング
- **高精度:** CosyVoice 1.0比で発音精度30-50%向上
- **高品質:** MOS評価 5.53（CosyVoice 1.0は5.4）
- **パラメータ数:** 500M
- **サンプリングレート:** 24000 Hz
- **フレームレート:** 25 Hz

**対応言語:** 中国語、英語、日本語、韓国語、中国語方言（広東語、四川語、上海語など）

**対応モード:**
- ゼロショット音声クローニング（3秒のサンプルから音声再現）
- クロスリンガル合成（言語を超えた音声変換）
- Instruct mode（感情・方言制御）
- きめ細かい制御（笑い、息継ぎなど）

## 重要なコマンド

### 環境構築

```bash
# 依存パッケージのインストール
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# sox互換性問題がある場合
# Ubuntu:
sudo apt-get install sox libsox-dev
# CentOS:
sudo yum install sox sox-devel
```

### モデルのダウンロード

```python
# Pythonでのモデルダウンロード
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

```bash
# Git LFSでのダウンロード
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
```

### Webデモの起動

```bash
# CosyVoice2-0.5Bでのデモ起動
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

### トレーニング

```bash
# LibriTTSデータセットでのトレーニング例
cd examples/libritts/cosyvoice2
# run.shを編集してdata_dir、pretrained_model_dirなどを設定
bash run.sh

# トレーニングエンジン: torch_ddp または deepspeed
# モデル: llm, flow, hifigan を順番にトレーニング
```

### デプロイメント

```bash
# FastAPIサーバー（REST API）
cd runtime/python/fastapi
python3 server.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B

# gRPCサーバー（高効率）
cd runtime/python/grpc
python3 server.py --port 50000 --max_conc 4 --model_dir pretrained_models/CosyVoice2-0.5B

# Triton + TensorRT-LLM（最高性能、NVIDIA GPUのみ）
cd runtime/triton_trtllm
docker compose up -d
```

### モデルのエクスポート

```bash
# 推論高速化のためのJIT/ONNXエクスポート
python cosyvoice/bin/export_jit.py --model_dir pretrained_models/CosyVoice2-0.5B
python cosyvoice/bin/export_onnx.py --model_dir pretrained_models/CosyVoice2-0.5B
```

### vLLMを使った推論

```bash
# vLLM環境の準備（既存環境を保護）
conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm
pip install vllm==v0.9.0 transformers==4.51.3

# vLLM推論の実行
python vllm_example.py
```

## アーキテクチャ

### コアコンポーネントの役割

CosyVoice2は3つの主要モジュールから構成されています：

#### 1. LLM (Qwen2LM) - テキストから離散音声トークンへ

**役割:** テキスト表現を離散的な音声トークンシーケンスに変換

**実装:**
- ファイル: `cosyvoice/llm/llm.py` (Qwen2LM クラス)
- バックボーン: Qwen2ForCausalLM（事前学習済み）
- 音声トークン数: 6,561
- テキスト:音声トークン比率: 5:15

**処理フロー:**
1. Qwen2のembed_tokensで直接テキスト埋め込み
2. 話者embedding（Campplus）統合
3. プロンプト音声トークンの組み込み
4. Bidirectional Streaming: テキストと音声を5:15の比率で混合
5. fill_token (token_id=6563) でストリーミング制御
6. 自己回帰生成: [SOS, embedding, text_tokens, task_id, prompt_tokens] → speech_tokens
7. KVキャッシュで逐次生成
8. RAS (Repetition Aware Sampling) で繰り返しを抑制

**vLLM統合:**
- バッチ推論の大幅高速化（4倍）
- 動的バッチング
- PagedAttention によるメモリ効率化

#### 2. Flow (CausalMaskedDiffWithXvec) - 音声トークンから音響特徴へ

**役割:** 離散トークンを連続的なメルスペクトログラムに変換

**実装:**
- ファイル: `cosyvoice/flow/flow.py`
- ファイル: `cosyvoice/flow/flow_matching.py`
- アーキテクチャ: CausalMaskedDiffWithXvec（因果的、ストリーミング対応）

**処理フロー:**
1. 音声トークン → Embedding (512次元)
2. UpsampleConformerEncoder (6ブロック):
   - 2倍アップサンプリング (token_mel_ratio=2)
   - static_chunk_size=25 でチャンク単位処理
   - pre_lookahead_len=3 で先読みコンテキスト
   - 因果的マスキング（過去のみ参照）
3. Conditional Flow Matching (CFM):
   - 条件: プロンプトメル + 話者embedding
   - ODE Solver: Euler法、10ステップ
   - t_scheduler: cosine
   - 推論時CFG (Classifier-Free Guidance)
4. 出力: メルスペクトログラム (80次元 × Tフレーム)

**ストリーミング対応:**
- チャンクごとの処理が可能
- キャッシュメカニズムで連続性確保
- オーバーラップ処理でスムーズな接続

#### 3. HiFiGAN - メルスペクトログラムから音声波形へ

**役割:** メルスペクトログラムを高品質な音声波形に変換

**実装:**
- ファイル: `cosyvoice/hifigan/generator.py`
- ファイル: `cosyvoice/hifigan/hifigan.py`
- アーキテクチャ: HiFTGenerator (HiFi-GAN + F0予測)

**処理フロー:**
1. F0予測器 (ConvRNNF0Predictor) → ピッチ情報抽出
2. NSF (Neural Source Filter):
   - 調波成分生成（8 harmonics）
   - ノイズ成分生成
3. アップサンプリング: [8, 5, 3] → 120倍 (24000 Hz)
4. ResBlocks: 残差ブロックで波形精緻化
5. iSTFT: スペクトル領域での最終処理

### データフロー

```
テキスト入力
    ↓
[LLM Module: cosyvoice/llm/llm.py - Qwen2LM]
    - Qwen2 Text Embedding
    - Speaker Embedding統合
    - Bidirectional Streaming (5:15混合)
    - 自己回帰トークン生成 (RAS)
    - KVキャッシュ + fill_token制御
    ↓
離散音声トークン (6,561語彙、例: [432, 1203, 2891, ...])
    ↓
[Flow Module: cosyvoice/flow/flow.py - CausalMaskedDiffWithXvec]
    - Token Embedding (512次元)
    - UpsampleConformerEncoder (2倍アップサンプリング)
    - static_chunk_size=25、因果的マスキング
    - Conditional Flow Matching (CFM、Euler法)
    ↓
メルスペクトログラム (80次元 × Tフレーム)
    ↓
[HiFiGAN Module: cosyvoice/hifigan/ - HiFTGenerator]
    - F0予測 (ConvRNNF0Predictor)
    - NSF (調波+ノイズ)
    - Upsampling [8, 5, 3] → 120倍
    - ResBlocks + iSTFT
    ↓
音声波形出力 (24000 Hz)
```

### モジュール構造

```
cosyvoice/
├── llm/              # Qwen2LM言語モデル
│   └── llm.py        # Qwen2LM実装（262-612行目）
├── flow/             # Flow Matching (音響特徴生成)
│   ├── flow.py       # CausalMaskedDiffWithXvec (151-282行目)
│   └── flow_matching.py  # Conditional CFM
├── hifigan/          # HiFiGAN vocoder (音声波形生成)
│   ├── generator.py  # HiFTGenerator
│   └── hifigan.py    # HiFiGAN統合
├── transformer/      # Conformer/Transformer実装
│   ├── encoder.py    # ConformerEncoder
│   └── upsample_encoder.py  # UpsampleConformerEncoder
├── tokenizer/        # テキストトークナイザー
│   └── tokenizer.py  # Qwen2Tokenizer統合
├── cli/              # ユーザー向けAPI
│   ├── cosyvoice.py  # CosyVoice2クラス
│   ├── frontend.py   # テキスト正規化、特徴抽出
│   └── model.py      # CosyVoice2Model統合、推論ロジック
├── dataset/          # データローディング、Parquet処理
├── utils/            # 共通ユーティリティ
├── bin/              # トレーニング・エクスポートスクリプト
│   └── train.py      # 統合トレーニングスクリプト
└── vllm/             # vLLM統合（高速化）
    └── cosyvoice2.py # vLLM対応モデル
```

### ストリーミング推論のメカニズム

CosyVoice2の双方向ストリーミングは、超低遅延（150ms以下）を実現する重要な機能です。

#### 1. LLMレベルのストリーミング

**Bidirectional Streaming:**
- テキストトークンと音声トークンを5:15の比率で混合トレーニング
- テキスト生成→音声生成を同時進行可能
- fill_token (token_id=6563) でチャンク境界を制御

**KVキャッシュ:**
- 過去のKey/Valueを保存
- 逐次生成時の計算量削減
- メモリ効率的なattention

**実装:** `cosyvoice/llm/llm.py` - Qwen2LMクラス

#### 2. Flowレベルのストリーミング

**因果的処理:**
- CausalMaskedDiffWithXvec: 過去のフレームのみ参照
- static_chunk_size=25: 25フレームごとに処理
- pre_lookahead_len=3: 3フレームの先読み

**チャンク管理:**
- 各チャンクは独立処理可能
- キャッシュメカニズムで連続性確保

**実装:** `cosyvoice/flow/flow.py` (151-282行目)

#### 3. 推論レベルのチャンク管理

**パラメータ (cosyvoice/cli/model.py):**
- `token_min_hop_len`: 最小チャンクサイズ (100トークン)
- `token_max_hop_len`: 最大チャンクサイズ (200トークン)
- `token_overlap_len`: オーバーラップ (20トークン)
- `mel_overlap_len`: メルオーバーラップ（20フレーム）

**処理フロー:**
1. LLMが100-200トークン生成したらチャンク開始
2. 20トークンのオーバーラップで前チャンクと連結
3. Flow/HiFiGANで音声化
4. mel/waveformキャッシュでスムーズな接続

#### 4. パフォーマンス指標

**First Chunk Latency:**
- 目標: 150ms以下
- 実測 (Triton TensorRT-LLM、L20 GPU): 189ms (use_spk2info_cache=True)

**RTF (Real-Time Factor):**
- HuggingFace実装: 0.0947 (batch=8)
- TensorRT-LLM: 0.0418 (batch=8) → **2.3倍高速化**
- vLLM: さらに高速化（動的バッチング）

## トレーニングパイプライン

### データ準備

**場所:** `examples/libritts/cosyvoice2/run.sh`

```bash
# Stage -1: データダウンロード（LibriTTS、MagicData等）
local/download_and_untar.sh ${data_dir} ${data_url} ${part}

# Stage 0: wav.scp/text/utt2spk/spk2utt生成
python local/prepare_data.py --src_dir $data_dir/LibriTTS/train --des_dir data/train

# Stage 1: 話者embeddings抽出（Campplus）
tools/extract_embedding.py --dir data/train --onnx_path pretrained_models/CosyVoice2-0.5B/campplus.onnx

# Stage 2: 音声トークン抽出（Speech Tokenizer v2）
tools/extract_speech_token.py --dir data/train --onnx_path pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx

# Stage 3: Parquetフォーマット変換
tools/make_parquet_list.py \
    --num_utts_per_parquet 1000 \
    --num_processes 10 \
    --src_dir data/train \
    --des_dir data/train/parquet
```

**データ構造:**
- wav.scp: 音声ファイルパス
- text: テキストトランスクリプション
- utt2spk: 発話→話者マッピング
- spk2utt: 話者→発話マッピング
- utt2embedding.pt: 発話ごとの話者embedding
- spk2embedding.pt: 話者ごとの平均embedding
- utt2speech_token.pt: 発話ごとの音声トークン

### モデルトレーニング

**場所:** `examples/libritts/cosyvoice2/run.sh` (Stage 5)

```bash
# 3つのモデルを順番にトレーニング: llm → flow → hifigan
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=4
job_id=1986
train_engine=torch_ddp  # または deepspeed

cat data/train-*/parquet/data.list > data/train.data.list
cat data/dev-*/parquet/data.list > data/dev.data.list

for model in llm flow hifigan; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --model $model \
      --checkpoint pretrained_models/CosyVoice2-0.5B/$model.pt \
      --model_dir exp/cosyvoice2/$model/$train_engine \
      --tensorboard_dir tensorboard/cosyvoice2/$model/$train_engine \
      --ddp.dist_backend nccl \
      --num_workers 2 \
      --prefetch 100 \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
done
```

**トレーニングエンジン:**
- `torch_ddp`: DistributedDataParallel（標準、全プラットフォーム対応）
- `deepspeed`: DeepSpeed Stage 2最適化（Linux限定、メモリ効率向上）

**トレーニングパラメータ (cosyvoice2.yaml):**
- llm: Qwen2LM、事前学習済みから継続学習
- flow: CausalMaskedDiffWithXvec、因果的Flow Matching
- hifigan: HiFTGenerator、24kHz対応

### モデル平均化とエクスポート

```bash
# Stage 6: モデル平均化（最後の5チェックポイント）
average_num=5
for model in llm flow hifigan; do
    python cosyvoice/bin/average_model.py \
      --dst_model exp/cosyvoice2/$model/$train_engine/${model}.pt \
      --src_path exp/cosyvoice2/$model/$train_engine \
      --num ${average_num} \
      --val_best
done

# Stage 7: JIT/ONNXエクスポート（推論高速化）
python cosyvoice/bin/export_jit.py --model_dir pretrained_models/CosyVoice2-0.5B
python cosyvoice/bin/export_onnx.py --model_dir pretrained_models/CosyVoice2-0.5B
```

### GRPO (強化学習) トレーニング

**目的:** Reinforcement Learningで発音精度とMOS評価を向上

**場所:** `examples/grpo/cosyvoice2/`

**パイプライン:**

```bash
# Stage -1: HuggingFace形式への変換
python pretrained_to_huggingface.py \
    --input_dir pretrained_models/CosyVoice2-0.5B \
    --output_dir huggingface_models/CosyVoice2-0.5B

# Stage 0: データ準備（verl形式Parquet、AIShell-3使用）
python prepare_data.py

# Stage 1: Reward Serverの起動（Token2wav + ASR）
python token2wav_asr_server.py \
    --model_dir ./huggingface_models/CosyVoice2-0.5B \
    --port 8888

# Stage 2: GRPOトレーニング
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_k=25 \
    custom_reward_function.name=compute_score \
    custom_reward_function.server_url=http://localhost:8888 \
    data.train_files=data/train.parquet \
    data.val_files=data/val.parquet

# Stage 3-4: モデルマージとSeed-TTSベンチマーク評価
# Stage 5: CosyVoiceフォーマットへの変換
python huggingface_to_pretrained.py \
    --input_dir fsdp_models/final \
    --output_dir pretrained_models/CosyVoice2-0.5B-GRPO
```

**Reward関数:**
- WER (Word Error Rate) ベース
- 低いWER = 高い報酬 = 発音精度向上
- ASRモデルで合成音声を認識し、元のテキストと比較

**GRPOアルゴリズム:**
- Group Relative Policy Optimization
- vLLMでロールアウト（高速サンプリング）
- temperature=0.8、top_k=25 で多様性確保
- Advantage推定でポリシー更新

## デプロイメント方法

### 1. FastAPI (REST API)

**場所:** `runtime/python/fastapi/`

**特徴:**
- RESTful API
- StreamingResponseでチャンク返却
- シンプルなHTTPクライアント

**エンドポイント:**
- `/inference_zero_shot`: ゼロショット音声クローニング
- `/inference_cross_lingual`: クロスリンガル合成
- `/inference_instruct2`: CosyVoice2のInstruct mode

**起動:**
```bash
cd runtime/python/fastapi
python3 server.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

**クライアント例:**
```bash
python3 client.py --port 50000 --mode zero_shot \
    --tts_text "你好，很高兴认识你" \
    --prompt_text "希望你以后能够做的比我还好呦" \
    --prompt_wav asset/zero_shot_prompt.wav
```

**Python APIコール:**
```python
import requests

response = requests.post(
    'http://localhost:50000/inference_zero_shot',
    json={
        'tts_text': '你好，很高兴认识你',
        'prompt_text': '希望你以后能够做的比我还好呦',
        'prompt_wav': 'path/to/prompt.wav',
        'stream': False
    }
)
```

### 2. gRPC (高効率バイナリプロトコル)

**場所:** `runtime/python/grpc/`

**特徴:**
- バイナリプロトコル（FastAPIより高速）
- ThreadPoolExecutor: 並列リクエスト処理 (max_conc指定)
- ストリーミングレスポンス対応
- 低遅延・高スループット

**起動:**
```bash
cd runtime/python/grpc
python3 server.py \
    --port 50000 \
    --max_conc 4 \
    --model_dir pretrained_models/CosyVoice2-0.5B
```

**クライアント例:**
```bash
python3 client.py --port 50000 --mode zero_shot \
    --tts_text "你好，很高兴认识你" \
    --prompt_text "希望你以后能够做的比我还好呦" \
    --prompt_wav asset/zero_shot_prompt.wav
```

**Protocol Buffers定義:** `runtime/python/grpc/cosyvoice.proto`

### 3. Triton + TensorRT-LLM (最高性能)

**場所:** `runtime/triton_trtllm/`

**特徴:**
- NVIDIA Triton Inference Server
- TensorRT-LLM: Qwen2LLMを4倍高速化
- Decoupled mode: ストリーミング/オフライン切り替え
- 最高のスループットとレイテンシ

**アーキテクチャ:**
```
Triton Inference Server
├── audio_tokenizer      # Speech tokenizer (ONNX)
├── speaker_embedding    # Campplus (ONNX)
├── tensorrt_llm         # Qwen2 TensorRT-LLM化
├── cosyvoice2           # オーケストレーションモデル
└── token2wav            # Flow + HiFiGAN (PyTorch)
```

**パフォーマンス (L20 GPU):**
- HuggingFace LLM: RTF 0.0947 (batch=8)
- TensorRT-LLM: RTF 0.0418 (batch=8) → **2.3倍高速化**
- ストリーミングFirst Chunk Latency: 189ms
- オフラインRTF: 0.0418（高スループット）

**セットアップ:**
```bash
cd runtime/triton_trtllm

# Stage 0: モデルダウンロード
# Stage 1: TensorRT-LLMエンジン構築
bash scripts/build_trtllm.sh

# Stage 2: Tritonモデルリポジトリ作成
python scripts/fill_template.py

# Stage 3: Tritonサーバー起動
docker compose up -d

# Stage 4-6: ベンチマーク実行
python streaming_inference.py  # ストリーミングベンチマーク
python offline_inference.py    # オフラインベンチマーク
```

**クライアント例:**
```bash
# gRPCクライアント
python client_grpc.py \
    --text "你好，很高兴认识你" \
    --prompt_text "希望你以后能够做的比我还好呦" \
    --prompt_wav asset/zero_shot_prompt.wav \
    --output output.wav

# HTTPクライアント
python client_http.py \
    --text "你好，很高兴认识你" \
    --prompt_text "希望你以后能够做的比我还好呦" \
    --prompt_wav asset/zero_shot_prompt.wav \
    --output output.wav
```

**パフォーマンスチューニング:**
- `max_batch_size`: バッチサイズ調整（スループット向上）
- `max_queue_delay_microseconds`: キューイング時間（レイテンシ調整）
- `instance_count`: モデルインスタンス数（並列処理）

### デプロイメント方法の選択

| 方法 | 用途 | レイテンシ | スループット | 実装難易度 |
|------|------|------------|--------------|------------|
| FastAPI | プロトタイピング、小規模 | 中 | 中 | 低 |
| gRPC | 本番環境、中規模 | 低 | 高 | 中 |
| Triton+TensorRT-LLM | 本番環境、大規模、最高性能 | 最低 | 最高 | 高 |

## vLLM統合

vLLMは、CosyVoice2のQwen2LLMを高速化する重要な最適化です。

**要件:**
- `vllm==v0.9.0`
- `torch==2.7.0`
- NVIDIA GPU必須

**インストール:**
```bash
# 新しいenvでインストール（既存環境を保護）
conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm
pip install vllm==v0.9.0 transformers==4.51.3 -i https://mirrors.aliyun.com/pypi/simple/
```

**使用例:**
```python
# vllm_example.py
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# vLLMを有効化してモデル初期化
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_vllm=True)

# 推論（vLLMが自動的に使用される）
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot(
    '你好，很高兴认识你',
    '希望你以后能够做的比我还好呦',
    prompt_speech_16k,
    stream=False)):
    torchaudio.save(f'output_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

**vLLMの最適化:**
- **PagedAttention:** KVキャッシュのメモリ効率化
- **動的バッチング:** 複数リクエストの自動バッチ化
- **Continuous Batching:** リクエストの動的追加/削除
- **Optimized CUDA Kernels:** カスタムCUDAカーネル

**パフォーマンス向上:**
- HuggingFace比で約4倍高速化
- メモリ使用量削減
- スループット大幅向上

**実装:** `cosyvoice/vllm/cosyvoice2.py`

## 基本的な使用例

### CosyVoice2の初期化

```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# モデル初期化
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,      # JIT最適化（オプション）
    load_trt=False,      # TensorRT最適化（オプション、NVIDIA GPUのみ）
    load_vllm=False,     # vLLM統合（オプション、高速化）
    fp16=False           # FP16推論（オプション、メモリ削減）
)
```

### ゼロショット音声合成

```python
# プロンプト音声の読み込み（3-30秒推奨）
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# ゼロショット推論
for i, j in enumerate(cosyvoice.inference_zero_shot(
    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    '希望你以后能够做的比我还好呦。',  # プロンプトテキスト
    prompt_speech_16k,                    # プロンプト音声
    stream=False)):                       # ストリーミング無効
    torchaudio.save(f'zero_shot_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### ストリーミング推論

```python
# ストリーミングモード（低遅延）
for i, j in enumerate(cosyvoice.inference_zero_shot(
    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    '希望你以后能够做的比我还好呦。',
    prompt_speech_16k,
    stream=True)):  # ストリーミング有効
    # チャンクごとに音声が生成される
    torchaudio.save(f'streaming_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
    print(f"Chunk {i}: {j['tts_speech'].shape}")
```

### 話者登録と再利用

```python
# 話者を登録（後で再利用可能）
success = cosyvoice.add_zero_shot_spk(
    '希望你以后能够做的比我还好呦。',
    prompt_speech_16k,
    'my_speaker_001'  # 話者ID
)

# 登録した話者で推論
for i, j in enumerate(cosyvoice.inference_zero_shot(
    '收到好友从远方寄来的生日礼物。',
    '',  # プロンプトテキスト不要
    '',  # プロンプト音声不要
    zero_shot_spk_id='my_speaker_001',  # 登録した話者ID使用
    stream=False)):
    torchaudio.save(f'registered_spk_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)

# 話者情報を保存（永続化）
cosyvoice.save_spkinfo()  # spk2info.ptに保存
```

### Instructモード（感情・方言制御）

```python
# 方言制御
for i, j in enumerate(cosyvoice.inference_instruct2(
    '收到好友从远方寄来的生日礼物。',
    '用四川话说这句话',  # 指示: 四川語で話す
    prompt_speech_16k,
    stream=False)):
    torchaudio.save(f'sichuan_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)

# 感情制御
for i, j in enumerate(cosyvoice.inference_instruct2(
    '今天真是太开心了！',
    '用开心的语气说',  # 指示: 嬉しい口調で
    prompt_speech_16k,
    stream=False)):
    torchaudio.save(f'happy_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### きめ細かい制御（笑い、息継ぎなど）

```python
# 特殊トークンでの制御
# サポートされる制御: cosyvoice/tokenizer/tokenizer.py#L248 参照
# [laughter], [breath], <strong></strong> など

for i, j in enumerate(cosyvoice.inference_cross_lingual(
    '在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。',
    prompt_speech_16k,
    stream=False)):
    torchaudio.save(f'fine_grained_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### クロスリンガル合成

```python
# 日本語プロンプト → 英語合成
prompt_speech_16k = load_wav('./asset/japanese_prompt.wav', 16000)

for i, j in enumerate(cosyvoice.inference_cross_lingual(
    '<|en|>Hello, nice to meet you. I am very happy to see you today.',
    prompt_speech_16k,
    stream=False)):
    torchaudio.save(f'cross_lingual_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### バイストリーム入力（LLM統合）

```python
# テキストジェネレーターからの入力（LLMの出力を逐次処理）
def text_generator():
    yield '收到好友从远方寄来的生日礼物，'
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐，'
    yield '笑容如花儿般绽放。'

for i, j in enumerate(cosyvoice.inference_zero_shot(
    text_generator(),  # ジェネレーターを直接渡す
    '希望你以后能够做的比我还好呦。',
    prompt_speech_16k,
    stream=False)):
    torchaudio.save(f'bistream_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

## 重要な設定ファイル

### トレーニング設定

**メイン設定:**
- `examples/libritts/cosyvoice2/conf/cosyvoice2.yaml`: CosyVoice2アーキテクチャ定義
- `examples/libritts/cosyvoice2/conf/ds_stage2.json`: DeepSpeed Stage 2設定

**cosyvoice2.yaml の主要パラメータ:**
```yaml
# LLM設定
llm:
  llm_type: qwen2  # Qwen2LM使用
  llm_input_size: 896
  llm_output_size: 896
  speech_token_size: 6561
  mix_ratio: [5, 15]  # テキスト:音声 = 5:15

# Flow設定
flow:
  flow_type: causal_masked_diff  # 因果的Flow Matching
  token_mel_ratio: 2  # 2倍アップサンプリング
  static_chunk_size: 25
  pre_lookahead_len: 3

# HiFiGAN設定
hifigan:
  resblock_kernel_sizes: [3, 7, 11]
  upsample_rates: [8, 5, 3]  # 24kHz対応
  upsample_initial_channel: 512

# サンプリングレート
sample_rate: 24000
```

### モデル設定（ダウンロードしたモデルディレクトリ内）

**主要ファイル:**
- `pretrained_models/CosyVoice2-0.5B/cosyvoice.yaml`: モデルアーキテクチャ定義
- `pretrained_models/CosyVoice2-0.5B/llm.pt`: Qwen2LM チェックポイント
- `pretrained_models/CosyVoice2-0.5B/flow.pt`: Flow モデルチェックポイント
- `pretrained_models/CosyVoice2-0.5B/hift.pt`: HiFiGAN チェックポイント
- `pretrained_models/CosyVoice2-0.5B/spk2info.pt`: 話者情報キャッシュ
- `pretrained_models/CosyVoice2-0.5B/campplus.onnx`: 話者embeddingモデル
- `pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx`: 音声トークナイザー

### JIT/TensorRT最適化モデル（オプション）

**JIT (TorchScript) モデル:**
- `llm.text_encoder.fp32.zip` / `llm.text_encoder.fp16.zip`
- `llm.llm.fp32.zip` / `llm.llm.fp16.zip`
- `flow.encoder.fp32.zip` / `flow.encoder.fp16.zip`

**TensorRT モデル:**
- `flow.decoder.estimator.fp32.mygpu.plan`
- `flow.decoder.estimator.fp16.mygpu.plan`

## テキスト正規化

CosyVoice2は2つのテキスト正規化方法をサポートしています。

### WeText（デフォルト）

**特徴:**
- Pythonベース
- クロスプラットフォーム（Linux/macOS/Windows）
- 依存関係なし

**使用:** 自動（ttsfrdが利用不可の場合）

### ttsfrd（オプション、高性能）

**特徴:**
- C++実装、高速
- Linux専用
- 中国語正規化に最適化

**インストール:**
```bash
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

**使用:** インストール後、自動的に優先使用

## 依存関係の重要事項

### Python & PyTorch

- **Python:** 3.10
- **PyTorch:** 2.3.1（標準）、2.7.0（vLLM使用時）
- **CUDA:** 12.1

### 主要パッケージ

- **transformers:** 4.51.3（Qwen2サポート）
- **conformer:** 0.3.2
- **diffusers:** 0.29.0
- **onnxruntime-gpu:** 1.18.0（Linux）
- **onnxruntime:** 1.18.0（macOS/Windows）
- **vllm:** v0.9.0（オプション、高速化）

### プラットフォーム制限

- **DeepSpeed:** Linux専用（macOS/Windowsでは使用不可）
- **TensorRT:** NVIDIA GPU専用
- **onnxruntime-gpu:** Linux専用
- **ttsfrd:** Linux専用

### 推奨環境

**最小要件:**
- GPU: NVIDIA GPU（6GB VRAM以上）
- RAM: 16GB
- ストレージ: 10GB以上

**推奨要件:**
- GPU: NVIDIA GPU（12GB VRAM以上、A100/L20推奨）
- RAM: 32GB以上
- ストレージ: 50GB以上

**トレーニング要件:**
- GPU: 複数のNVIDIA GPU（各16GB VRAM以上）
- RAM: 64GB以上
- ストレージ: 500GB以上（データセット次第）

## 注意事項

### 必須事項

1. **サブモジュール初期化:**
   ```bash
   git submodule update --init --recursive
   ```
   `third_party/Matcha-TTS`が必要

2. **sys.path設定:**
   ```python
   sys.path.append('third_party/Matcha-TTS')
   ```
   全てのPythonスクリプトで必要

### モデル使用

3. **モデルサイズ:** CosyVoice2-0.5Bは約3-4GB。十分なディスク容量を確保
4. **プロンプト音声:** 3-30秒を推奨。長すぎると品質低下、短すぎると話者特徴が不十分
5. **text_frontend:** 本家デモサイトの結果を再現する場合、`text_frontend=False`を指定

### プラットフォーム

6. **macOS/Windows:** DeepSpeed非対応。`train_engine=torch_ddp`を使用
7. **CPU推論:** 可能だが非常に遅い。GPU使用を強く推奨

### ストリーミング

8. **ストリーミング推論:** 双方向ストリーミングはCosyVoice2の主要機能
9. **First Chunk Latency:** 150ms目標、実測189ms（Triton TensorRT-LLM）
10. **チャンク管理:** オーバーラップ処理により、音声の連続性を確保

### デプロイメント

11. **Dockerデプロイ:** `runtime/python/`にDockerfileあり。NVIDIA runtimeが必要
12. **Triton TensorRT-LLM:** セットアップが複雑だが、最高性能を提供

### vLLM

13. **vLLM環境:** 既存環境を破壊しないよう、新しいconda envで使用
14. **PyTorchバージョン:** vLLM v0.9.0はtorch 2.7.0が必要

### GRPO

15. **GRPO環境:** HuggingFace形式への変換が必要
16. **Reward Server:** ASRモデルが必要（WhisperなどWhisper）

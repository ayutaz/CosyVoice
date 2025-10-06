# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CosyVoice 2.0 - スケーラブルなストリーミング音声合成システム

## プロジェクト概要

CosyVoice 2.0は、Alibabaが開発した大規模言語モデル(LLM)ベースのText-to-Speech(TTS)システムです。Qwen2をバックボーンとし、ストリーミング推論に対応した次世代音声合成モデルです。

### 主要な特徴

- **超低レイテンシ**: 150msの初回音声生成
- **ストリーミング対応**: 双方向ストリーミング(Bi-stream)推論
- **高精度**: CosyVoice 1.0と比較して発音エラーを30-50%削減
- **多言語対応**: 中国語、英語、日本語、韓国語、中国語方言(広東語、四川語、上海語、天津語、武漢語など)
- **ゼロショット音声クローニング**: クロスリンガル・コードスイッチングにも対応
- **高品質**: MOS評価スコア 5.4 → 5.53に向上
- **サンプリングレート**: 24000 Hz
- **モデルサイズ**: 0.5Bパラメータ

### 技術的進化

CosyVoice 2.0は従来バージョンから以下の点が改善されています:

1. **Finite-Scalar Quantization**: 改善されたcodebook利用率
2. **Qwen2バックボーン**: 事前学習済みLLMを直接使用
3. **Chunk-Aware Causal Flow Matching**: ストリーミング・非ストリーミング両対応
4. **大規模多言語データセット**: 人間レベルの自然さを実現

## アーキテクチャ

CosyVoice 2.0は3つの主要コンポーネントで構成されています:

### 1. Qwen2LM (Large Language Model)
**場所**: `cosyvoice/llm/llm.py`

#### アーキテクチャ詳細
- **Backbone**: Qwen2ForCausalLM (事前学習済み大規模言語モデル)
- **Speech Token Vocabulary**: 6561トークン
- **Special Tokens**:
  - `sos_eos` (token_id: 0): 系列開始/終了
  - `task_id` (token_id: 1): タスク識別子
  - `fill_token` (token_id: 2): ストリーミング用フィルトークン
- **Output Size**: 896次元

#### 処理フロー
```
テキスト → Qwen2 Embeddings → LLM Encoder
                                    ↓
                              Concat (SOS + Text + Task ID)
                                    ↓
                            Speech Token Generation
                                    ↓
                        Output: 6561語彙の音声トークン系列
```

#### Bi-stream Training
CosyVoice 2.0の革新的な機能:

- **Mix Ratio**: [5, 15] (テキスト5トークン : 音声15トークン)
- **Uni-stream Mode**: 従来型の一括処理
- **Bi-stream Mode**: テキストと音声を交互に処理
  - Fill token (`speech_token_size + 2`)で区切り
  - ストリーミング推論に最適化

```python
# Bi-stream例
# [SOS] → [Text1-5] → [Speech1-15] → [Fill] → [Text6-10] → [Speech16-30] → [Fill] → ...
```

#### vLLM統合
- **vLLM Version**: v0.9.0
- **Performance**: 4x高速化（TensorRT-LLM使用時）
- **Features**:
  - バッチ推論最適化
  - KVキャッシュ効率化
  - 並列デコーディング

### 2. CausalConditionalCFM (Flow Matching Model)
**場所**: `cosyvoice/flow/flow_matching.py`, `cosyvoice/flow/flow.py`

#### アーキテクチャ詳細
- **Encoder**: UpsampleConformerEncoder (6ブロック、512次元)
- **Decoder**: CausalConditionalDecoder (UNet風アーキテクチャ)
- **Input Frame Rate**: 25 token/s
- **Token-Mel Ratio**: 2 (1トークン = 2メルフレーム)
- **Static Chunk Size**: 25トークン (ストリーミング推論用)

#### Flow Matching特性
- **Solver**: Euler solver with cosine t-scheduler
- **Classifier-Free Guidance**:
  - Training CFG Rate: 0.2
  - Inference CFG Rate: 0.7
- **Sigma Min**: 1e-06

#### Causal Streaming
```
Input: 音声トークン系列 → Chunk分割 (25トークン単位)
         ↓
UpsampleConformerEncoder (因果的処理)
         ↓
CausalConditionalDecoder (チャンクごとに処理)
         ↓
Output: メル特徴量 (50フレーム/チャンク)
```

- **Pre-lookahead**: 3トークン先読み
- **Left Chunks**: -1 (全ての過去チャンクを使用)

### 3. HiFTGenerator (Vocoder)
**場所**: `cosyvoice/hifigan/generator.py`

#### 仕様
- **Base Channels**: 512
- **Harmonics**: 8
- **Upsampling Rates**: [8, 5, 3]
- **Total Upsampling**: 120x (480サンプル/フレーム → 24000 Hz)
- **ISTFT Parameters**:
  - n_fft: 16
  - hop_len: 4
- **F0 Predictor**: ConvRNNF0Predictor

#### ストリーミング最適化
- **Cache Mechanism**:
  - Mel Cache: 8フレーム
  - Source Cache: 3840サンプル (8 * 480)
- **Fade In/Out**: Hamming window によるスムージング

## 学習方法

### データ準備パイプライン

#### 前処理ステップ (`examples/libritts/cosyvoice2/run.sh`)

```bash
# Stage 0: データ準備
python local/prepare_data.py --src_dir $data_dir --des_dir data/$x

# Stage 1: Speaker Embedding抽出 (CAM++)
tools/extract_embedding.py --dir data/$x \
    --onnx_path pretrained_models/CosyVoice2-0.5B/campplus.onnx

# Stage 2: 音声トークン抽出 (Finite-Scalar Quantization)
tools/extract_speech_token.py --dir data/$x \
    --onnx_path pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx

# Stage 3: Parquetフォーマット変換
tools/make_parquet_list.py --num_utts_per_parquet 1000 \
    --num_processes 10 \
    --src_dir data/$x \
    --des_dir data/$x/parquet
```

#### データ処理パイプライン (`cosyvoice/dataset/processor.py`)

1. **Parquet Opener**: 効率的なデータロード
2. **Tokenization**: Qwen2 tokenizer使用
3. **Filtering**:
   - 最大長: 40960サンプル
   - 最小長: 100サンプル
   - トークン最大長: 200
4. **Resampling**: 24000 Hzに統一
5. **Mel-spectrogram抽出**:
   - n_fft: 1920
   - hop_size: 480
   - win_size: 1920
   - num_mels: 80
6. **Speaker Embedding解析**: L2正規化
7. **Shuffling**: シャッフルサイズ 1000
8. **Sorting**: ソートサイズ 500
9. **Dynamic Batching**: max_frames_in_batch=2000

### トレーニング

#### 3段階トレーニング戦略

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=4

# Stage 5: LLM → Flow → HiFiGAN を順次トレーニング
for model in llm flow hifigan; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=1986 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
        cosyvoice/bin/train.py \
        --train_engine torch_ddp \
        --config conf/cosyvoice2.yaml \
        --train_data data/train.data.list \
        --cv_data data/dev.data.list \
        --model $model \
        --checkpoint pretrained_models/CosyVoice2-0.5B/$model.pt \
        --use_amp \
        --deepspeed_config ./conf/ds_stage2.json
done
```

#### トレーニング設定 (`conf/cosyvoice2.yaml`)

**LLM/Flow Training**:
- Optimizer: Adam
- Learning Rate: 1e-5 (SFT)
- Scheduler: ConstantLR
- Gradient Clipping: 5
- Accumulation Steps: 2
- Loss: Label Smoothing Cross Entropy (smoothing=0, length_normalized=True)

**HiFiGAN Training**:
- Generator/Discriminator LR: 0.0002
- Scheduler: ConstantLR
- Accumulation: 1 (必須)
- Discriminators:
  - MultiPeriodDiscriminator
  - MultiResSpecDiscriminator

#### モデル平均化

```bash
# Stage 6: Top-5モデルの平均化
python cosyvoice/bin/average_model.py \
    --dst_model exp/cosyvoice2/$model/$model.pt \
    --src_path exp/cosyvoice2/$model \
    --num 5 \
    --val_best
```

### GRPO (Group Relative Policy Optimization)
**場所**: `examples/grpo/cosyvoice2/`

CosyVoice 2.0のポストトレーニング手法:

- **Reward Model**: ASRベースの音声品質評価
- **Loss**: DPO (Direct Preference Optimization)風
- **Chosen/Rejected Samples**: ペア学習
- **Logits計算**: 選択/拒否系列の対数確率

## 他のOSS TTSとの比較

| 特徴 | CosyVoice 2.0 | VITS/VITS2 | Coqui TTS | Bark | VALL-E |
|-----|--------------|-----------|-----------|------|---------|
| **アーキテクチャ** | LLM + Flow Matching | VAE + GAN | 多様 | GPT-based | Codec LM |
| **音声トークン** | Supervised (FSQ) | なし | 一部モデル | EnCodec | EnCodec |
| **ゼロショット** | ✓ 高品質 | △ | △ | ✓ | ✓ |
| **ストリーミング** | ✓ 150ms | ✗ | △ | ✗ | ✗ |
| **多言語** | ✓ 9言語+方言 | △ | ✓ | ✓ | △ |
| **モデルサイズ** | 500M | 数十M | 様々 | 1B+ | 数百M |
| **推論速度** | 高 (vLLM) | 高 | 中 | 低 | 中 |
| **事前学習LLM** | ✓ Qwen2 | ✗ | ✗ | ✓ GPT | ✗ |

### CosyVoice 2.0の技術的優位性

1. **Supervised Semantic Tokens (FSQ)**
   - Finite-Scalar Quantizationによる高効率
   - 従来のunsupervised tokens (EnCodec等)より高品質
   - Codebook利用率の改善

2. **Qwen2バックボーン統合**
   - 事前学習済みLLMの知識を活用
   - スケーリング効果が高い
   - In-context learning能力

3. **Chunk-Aware Causal Flow Matching**
   - ストリーミング・非ストリーミング両対応
   - 因果的推論による低レイテンシ
   - 品質の劣化なし

4. **双方向ストリーミング (Bi-stream)**
   - テキストと音声を交互に処理
   - Fill tokenによる柔軟な制御
   - TTS-LLM統合に最適

5. **vLLM統合**
   - 4x高速化 (TensorRT-LLM)
   - バッチ推論最適化
   - プロダクション対応

## 開発コマンド

### 環境セットアップ

```bash
# リポジトリクローン
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

# Conda環境作成
conda create -n cosyvoice python=3.10
conda activate cosyvoice
pip install -r requirements.txt

# Sox インストール (オプション)
# Ubuntu:
sudo apt-get install sox libsox-dev
# CentOS:
sudo yum install sox sox-devel
```

### モデルダウンロード

#### Python SDK
```python
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

#### Git Clone
```bash
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

### 基本的な推論

```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# モデルロード
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')

# ゼロショット音声クローニング
prompt_speech = load_wav('./asset/zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot(
    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    '希望你以后能够做的比我还好呦。',
    prompt_speech,
    stream=False
)):
    torchaudio.save(f'zero_shot_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)

# スピーカーIDの保存
cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech, 'my_speaker')
cosyvoice.save_spkinfo()

# 保存したスピーカーIDで推論
for i, j in enumerate(cosyvoice.inference_zero_shot(
    '新しいテキスト',
    '', '',
    zero_shot_spk_id='my_speaker'
)):
    torchaudio.save(f'output_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### 高度な機能

#### 感情・方言制御 (Instruct2)
```python
# 四川語で話す
for i, j in enumerate(cosyvoice.inference_instruct2(
    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。',
    '用四川话说这句话',
    prompt_speech
)):
    torchaudio.save(f'sichuan_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

#### Fine-grained Control
```python
# [laughter]タグ使用
for i, j in enumerate(cosyvoice.inference_cross_lingual(
    '在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。',
    prompt_speech
)):
    torchaudio.save(f'control_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

#### Bi-stream推論 (テキストジェネレータ統合)
```python
def text_generator():
    yield '収到好友从远方寄来的生日礼物，'
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐，'
    yield '笑容如花儿般绽放。'

for i, j in enumerate(cosyvoice.inference_zero_shot(
    text_generator(),
    '希望你以后能够做的比我还好呦。',
    prompt_speech,
    stream=False
)):
    torchaudio.save(f'bistream_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### ストリーミング推論

```python
# ストリーミングモード
for i, j in enumerate(cosyvoice.inference_zero_shot(
    'とても長いテキストをストリーミングで処理します...',
    '希望你以后能够做的比我还好呦。',
    prompt_speech,
    stream=True  # ストリーミング有効化
)):
    # 音声チャンクが逐次生成される (レイテンシ: ~150ms)
    torchaudio.save(f'stream_chunk_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### WebUI起動

```bash
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

### vLLM推論 (高速化)

```bash
# 専用環境作成 (torch 2.7.0必須)
conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm
pip install vllm==v0.9.0 transformers==4.51.3

# 推論実行
python vllm_example.py
```

**Python コード**:
```python
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_vllm=True  # vLLM有効化
)
```

### TensorRT-LLM推論 (4x高速化)

```bash
cd runtime/triton_trtllm
docker compose up -d
```

詳細: https://github.com/FunAudioLLM/CosyVoice/tree/main/runtime/triton_trtllm

### Docker デプロイ

```bash
cd runtime/python
docker build -t cosyvoice:v2.0 .

# gRPCサーバー
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v2.0 \
    /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/grpc && \
    python3 server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice2-0.5B && \
    sleep infinity"

# クライアント
cd runtime/python/grpc
python3 client.py --port 50000 --mode zero_shot

# FastAPIサーバー
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v2.0 \
    /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && \
    python3 server.py --port 50000 --model_dir iic/CosyVoice2-0.5B && \
    sleep infinity"

# クライアント
cd runtime/python/fastapi
python3 client.py --port 50000 --mode zero_shot
```

### トレーニング

```bash
cd examples/libritts/cosyvoice2

# データ準備 (Stage 0-3)
bash run.sh --stage 0 --stop-stage 3

# LLM/Flow/HiFiGAN トレーニング (Stage 5)
export CUDA_VISIBLE_DEVICES="0,1,2,3"
bash run.sh --stage 5 --stop-stage 5

# モデル平均化 (Stage 6)
bash run.sh --stage 6 --stop-stage 6
```

### エクスポート (推論最適化)

```bash
# JIT/ONNXエクスポート
python cosyvoice/bin/export_jit.py --model_dir pretrained_models/CosyVoice2-0.5B
python cosyvoice/bin/export_onnx.py --model_dir pretrained_models/CosyVoice2-0.5B
```

## 主要ファイル構造

```
cosyvoice/
├── llm/
│   └── llm.py                      # Qwen2LM, Qwen2Encoder
├── flow/
│   ├── flow.py                     # CausalMaskedDiffWithXvec
│   ├── flow_matching.py            # CausalConditionalCFM
│   └── decoder.py                  # CausalConditionalDecoder
├── hifigan/
│   ├── generator.py                # HiFTGenerator (24kHz vocoder)
│   └── discriminator.py            # MultipleDiscriminator
├── transformer/
│   ├── encoder.py                  # ConformerEncoder
│   └── upsample_encoder.py         # UpsampleConformerEncoder
├── cli/
│   ├── cosyvoice.py                # CosyVoice2 (メインAPI)
│   ├── frontend.py                 # CosyVoiceFrontEnd
│   └── model.py                    # CosyVoice2Model
├── dataset/
│   ├── dataset.py                  # Dataset
│   └── processor.py                # データパイプライン
├── tokenizer/
│   └── tokenizer.py                # get_qwen_tokenizer
├── vllm/
│   └── cosyvoice2.py               # vLLM統合
└── bin/
    ├── train.py                    # トレーニング
    ├── average_model.py            # モデル平均化
    ├── export_jit.py               # JITエクスポート
    └── export_onnx.py              # ONNXエクスポート

examples/
├── libritts/cosyvoice2/
│   ├── run.sh                      # トレーニングパイプライン
│   └── conf/cosyvoice2.yaml        # 設定ファイル
└── grpo/cosyvoice2/                # GRPOポストトレーニング

runtime/
├── python/
│   ├── grpc/                       # gRPCサーバー/クライアント
│   └── fastapi/                    # FastAPIサーバー/クライアント
└── triton_trtllm/                  # TensorRT-LLM推論

tools/
├── extract_embedding.py            # Speaker embedding抽出
├── extract_speech_token.py         # 音声トークン抽出
└── make_parquet_list.py            # Parquetデータ作成
```

## 重要な設定パラメータ

### cosyvoice2.yaml (`examples/libritts/cosyvoice2/conf/cosyvoice2.yaml`)

```yaml
# 基本パラメータ
sample_rate: 24000
llm_input_size: 896
llm_output_size: 896
spk_embed_dim: 192
token_frame_rate: 25
token_mel_ratio: 2

# ストリーミングパラメータ
chunk_size: 25                      # トークン/チャンク
num_decoding_left_chunks: -1        # 全ての過去チャンクを使用

# LLMパラメータ
speech_token_size: 6561
mix_ratio: [5, 15]                  # Bi-stream比率

# Flow Matchingパラメータ
cfm_params:
    sigma_min: 1e-06
    solver: 'euler'
    t_scheduler: 'cosine'
    training_cfg_rate: 0.2
    inference_cfg_rate: 0.7

# トレーニングパラメータ
train_conf:
    optim: adam
    optim_conf:
        lr: 1e-5
    scheduler: constantlr
    max_epoch: 200
    grad_clip: 5
    accum_grad: 2
```

## 推論モード

CosyVoice 2.0は以下の推論モードをサポートします:

### 1. Zero-shot 音声クローニング
任意の音声プロンプトで話者の声を再現:
```python
cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech)
```

### 2. Cross-lingual 合成
異なる言語間での音声クローニング:
```python
cosyvoice.inference_cross_lingual(tts_text, prompt_speech)
```

### 3. Instruct2 制御
感情・方言・スタイルの指示制御:
```python
cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech)
# 例: instruct_text = "用四川话说这句话" (四川語で話す)
```

### 4. Bi-stream推論
テキストジェネレータとの統合:
```python
cosyvoice.inference_zero_shot(text_generator(), prompt_text, prompt_speech)
```

### 5. Voice Conversion
音声変換 (継承機能):
```python
cosyvoice.inference_vc(source_speech, prompt_speech)
```

## ストリーミング vs 非ストリーミング推論

### 非ストリーミング (`stream=False`)
- **レイテンシ**: 全テキスト処理完了まで待機
- **品質**: 最高品質
- **用途**: オフライン生成、高品質要求

### ストリーミング (`stream=True`)
- **レイテンシ**: ~150ms (初回チャンク)
- **品質**: ほぼ無劣化
- **用途**: リアルタイム会話、インタラクティブアプリ

```python
# ストリーミング例
for chunk in cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech, stream=True):
    # チャンクごとに音声が生成される
    play_audio(chunk['tts_speech'])
```

## テキスト正規化

2つのバックエンドをサポート:

1. **ttsfrd** (推奨):
```bash
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

2. **WeTextProcessing** (フォールバック):
ttsfrdが利用できない場合に自動的に使用されます。

## 依存関係

主要ライブラリ:
- **PyTorch**: 2.0+ (vLLM使用時は2.7.0)
- **transformers**: 4.51.3 (Qwen2モデル)
- **vllm**: v0.9.0 (オプション、高速推論)
- **tensorrt**: 8.0+ (オプション、推論最適化)
- **onnxruntime**: embedding/tokenizer用
- **matcha-tts**: Flow Matchingベースライン

## ベストプラクティス

### 1. モデルロード最適化

```python
# 基本ロード
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')

# JIT最適化
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True)

# vLLM高速化
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_vllm=True)

# TensorRT最適化
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_trt=True)

# FP16推論
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', fp16=True)
```

### 2. ゼロショット音声クローニングのコツ

- **プロンプト音声**: 3-10秒を推奨
- **プロンプトテキスト**: 音声と正確に一致させる
- **音質**: クリーンな音声を使用
- **デモ再現**: `text_frontend=False`を指定

```python
# デモサイト結果の再現
for i, j in enumerate(cosyvoice.inference_zero_shot(
    tts_text, prompt_text, prompt_speech,
    text_frontend=False  # テキスト前処理を無効化
)):
    torchaudio.save(f'output_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### 3. トレーニングのステップ

1. **データ準備**: 高品質な多言語データセット
2. **LLMトレーニング**: まずQwen2LMを微調整
3. **Flowトレーニング**: CausalConditionalCFMをトレーニング
4. **HiFiGANトレーニング**: Vocoderを微調整
5. **モデル平均化**: Top-5モデルを平均化
6. **GRPO (オプション)**: ポストトレーニングで品質向上

### 4. プロダクション配置

- **低レイテンシ**: vLLM + TensorRT-LLM
- **高スループット**: Docker + gRPC/FastAPI
- **スケーラビリティ**: Triton Inference Server

### 5. エラーハンドリング

```python
# スピーカーID保存/ロード
try:
    cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech, 'speaker_id')
    cosyvoice.save_spkinfo()
except Exception as e:
    print(f"Speaker registration failed: {e}")

# ストリーミング推論のタイムアウト処理
import signal
signal.alarm(60)  # 60秒タイムアウト
try:
    for chunk in cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech, stream=True):
        process_chunk(chunk)
finally:
    signal.alarm(0)
```

## 既知の制限事項

1. **vLLM要件**:
   - vllm==v0.9.0が必須
   - torch==2.7.0必須
   - 専用Conda環境推奨

2. **ストリーミング推論**:
   - スピード変更(`speed`パラメータ)非対応
   - チャンクサイズ固定(25トークン)

3. **TensorRT**:
   - GPU専用
   - GPU固有の`.plan`ファイルが必要
   - 異なるGPUでは再ビルド必要

4. **メモリ要件**:
   - 最小: 8GB GPU VRAM
   - 推奨: 16GB+ GPU VRAM (vLLM使用時)

## トラブルシューティング

### vLLMインストールエラー
```bash
# torch 2.7.0を先にインストール
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm==v0.9.0
```

### ストリーミング推論が遅い
```python
# stream_scale_factorを調整 (cosyvoice/cli/model.py)
self.stream_scale_factor = 1.5  # デフォルト: 1
```

### TensorRT ビルドエラー
```bash
# 正しいCUDAバージョンを確認
nvcc --version
# TensorRTバージョン確認
python -c "import tensorrt; print(tensorrt.__version__)"
```

## 参考リンク

- **Paper**: https://arxiv.org/abs/2412.10117
- **Demos**: https://funaudiollm.github.io/cosyvoice2/
- **ModelScope**: https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B
- **HuggingFace**: https://huggingface.co/spaces/FunAudioLLM/CosyVoice2-0.5B
- **GitHub**: https://github.com/FunAudioLLM/CosyVoice

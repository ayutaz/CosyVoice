# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 現在のプロジェクト目標

**CosyVoice3の日本語対応**

Fun-CosyVoice 3.0は公式で9言語以上をサポートしているが、日本語の品質向上・最適化を行う。

### 環境情報
- **OS**: Windows 11
- **Python**: 3.10 (uv管理)
- **モデル**: Fun-CosyVoice3-0.5B-2512
- **モデルパス**: `pretrained_models/Fun-CosyVoice3-0.5B`（snapshot_downloadで自動取得）

---

## CosyVoice3 日本語サポート現状調査

### 公式言語サポート

CosyVoice3は9言語をサポート:
- 中国語 `<|zh|>`, 英語 `<|en|>`, **日本語 `<|ja|>`**, 韓国語 `<|ko|>`
- ドイツ語, スペイン語, フランス語, イタリア語, ロシア語
- 18以上の中国方言（広東語 `<|yue|>` 等）

### 日本語対応の課題

#### 1. テキストフロントエンド (`cosyvoice/cli/frontend.py`)

**現状の問題:**
```python
# text_normalize() の処理フロー
if contains_chinese(text):  # 中国語判定
    # 中国語正規化
else:
    # 英語として処理 ← 日本語もここに入る
    text = spell_out_number(text, self.inflect_parser)  # 英語数字読み上げ
```

- `contains_chinese()`: Unicode中国語文字のみ判定（日本語判定なし）
- 日本語テキストは「英語」パスで処理される
- 日本語数字→読み仮名変換なし（例: "100" → "ひゃく"）
- 日本語記号・句読点の正規化なし

#### 2. 分かち書き (`cosyvoice/utils/frontend_utils.py`)

```python
def split_paragraph(text, tokenize, lang="zh", ...):
    if lang == "zh":
        pounc = ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';']
    else:  # 英語扱い
        pounc = ['.', '?', '!', ';', ':']
```

- 日本語は`lang`パラメータで明示的にサポートされていない
- 日本語句読点（。、？！）は中国語設定で部分的に対応

#### 3. トークナイザー (`cosyvoice/tokenizer/tokenizer.py`)

**CosyVoice3Tokenizer の特殊トークン:**
- 中国語ピンイン: `[ā]`, `[zhē]`, `[iàn]` 等
- 英語CMU音素: `[AA]`, `[TH]`, `[IY]` 等
- **日本語音素トークン: なし**

言語タグ `<|ja|>` は定義済み（tokenizer.py:19）

#### 4. 話者情報

- 各モデルに `spk2info.pt` が含まれる
- Fun-CosyVoice3の日本語ネイティブ話者の有無は未確認
- ゼロショット音声クローンで日本語話者を追加可能

### 日本語対応に必要な作業

| 優先度 | 作業項目 | 対象ファイル |
|--------|----------|--------------|
| 高 | 日本語テキスト判定関数追加 | `frontend_utils.py` |
| 高 | 日本語テキスト正規化追加 | `frontend.py` |
| 高 | 日本語数字読み上げ実装 | `frontend.py` |
| 中 | 分かち書きの日本語対応 | `frontend_utils.py` |
| 中 | 日本語句読点処理の最適化 | `frontend_utils.py` |
| 低 | 日本語音素トークン追加 | `tokenizer.py` |
| 低 | 日本語話者ファインチューニング | training |

### 日本語推論の使い方（現状）

```python
from cosyvoice.cli.cosyvoice import AutoModel

cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# cross_lingual モードで日本語タグを使用
for output in cosyvoice.inference_cross_lingual(
    '<|ja|>こんにちは、今日はいい天気ですね。',
    './prompt.wav'
):
    audio = output['tts_speech']

# text_frontend=False でテキスト正規化をスキップ（推奨）
for output in cosyvoice.inference_cross_lingual(
    '<|ja|>こんにちは',
    './prompt.wav',
    text_frontend=False
):
    audio = output['tts_speech']
```

### 関連ファイル

| ファイル | 役割 |
|----------|------|
| `cosyvoice/cli/frontend.py` | テキスト正規化、音声エンコード |
| `cosyvoice/cli/cosyvoice.py` | CosyVoice3クラス、AutoModel |
| `cosyvoice/utils/frontend_utils.py` | 中国語判定、分かち書き |
| `cosyvoice/tokenizer/tokenizer.py` | 言語定義、トークナイザー |
| `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml` | モデル設定 |

---

## プロジェクト概要

CosyVoiceは大規模言語モデル(LLM)ベースの音声合成(TTS)システムです。3つのバージョンがあります：
- **Fun-CosyVoice 3.0**: 9言語以上、18以上の中国方言をサポート
- **CosyVoice 2.0**: ストリーミング対応、vLLM統合 (0.5Bモデル)
- **CosyVoice 1.0**: オリジナル 300Mパラメータモデル

## CosyVoice3 モデル一覧

| モデル名 | サイズ | リポジトリID | 特徴 |
|---------|--------|-------------|------|
| Fun-CosyVoice3-0.5B-2512 | 0.5B | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | 最新版、9言語+18方言 |
| Fun-CosyVoice3-0.5B-2512_RL | 0.5B | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512_RL` | RLHF版、高精度（CER 0.81%） |

**注意**: 0.5Bモデルのみ。1B以上の大規模モデルは公開されていない。

### モデルダウンロード

```python
from modelscope import snapshot_download

# 通常版
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# RLHF版（より高精度）
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512_RL', local_dir='pretrained_models/Fun-CosyVoice3-0.5B-RL')
```

## セットアップコマンド

```bash
# クローン（サブモジュール必須）
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
git submodule update --init --recursive

# 環境構築（Python 3.10必須）
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# モデルダウンロード
python -c "
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
"
```

## 実行コマンド

```bash
# 推論テスト
python example.py

# Web UI (Gradio)
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B

# vLLM高速化 (v0.9.0)
pip install vllm==v0.9.0 transformers==4.51.3
python vllm_example.py

# vLLM v0.11.0+
pip install vllm==v0.11.0 transformers==4.57.1
python vllm_example.py
```

## トレーニング

```bash
python cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --model pretrained_models/CosyVoice2-0.5B \
  --config examples/libritts/cosyvoice2/conf/cosyvoice2.yaml \
  --train_data /path/to/train_data \
  --cv_data /path/to/cv_data \
  --model_dir ./checkpoints
```

## デプロイ

```bash
# FastAPI
cd runtime/python/fastapi && python server.py --port 50000

# gRPC
cd runtime/python/grpc && python server.py --port 50000

# Docker
cd runtime/python && docker build -t cosyvoice:v1.0 .
```

## アーキテクチャ

CosyVoiceは3段階パイプライン:
1. **LLM**: テキスト→音声トークン生成
2. **Flow Matching**: トークン→メルスペクトログラム変換
3. **HiFi-GAN (HIFT)**: メルスペクトログラム→波形変換

### コアモジュール

| ディレクトリ | 役割 |
|-------------|------|
| `cosyvoice/cli/` | ユーザーAPI層。`AutoModel`ファクトリー、推論メソッド |
| `cosyvoice/llm/` | TransformerLM、音声トークン生成 |
| `cosyvoice/flow/` | Flow Matching、DiT、長さ調整器 |
| `cosyvoice/hifigan/` | ボコーダー（メル→波形） |
| `cosyvoice/transformer/` | Conformer、Transformer実装 |
| `cosyvoice/tokenizer/` | テキスト/音声トークン処理 |
| `cosyvoice/vllm/` | vLLM統合ラッパー |
| `cosyvoice/bin/` | トレーニング、エクスポートスクリプト |

### 主要エントリーポイント

- `cosyvoice/cli/cosyvoice.py`: `AutoModel()` - モデルバージョン自動検出
- `cosyvoice/cli/model.py`: 内部モデルラッパー、最適化ロード
- `cosyvoice/cli/frontend.py`: テキスト正規化、音声エンコード

## 推論モード

```python
from cosyvoice.cli.cosyvoice import AutoModel

cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')

# SFT（事前学習話者）
for output in cosyvoice.inference_sft('text', 'speaker_name'):
    audio = output['tts_speech']

# ゼロショット（音声クローン）
for output in cosyvoice.inference_zero_shot('text', 'prompt_text', 'prompt.wav'):
    audio = output['tts_speech']

# 多言語
for output in cosyvoice.inference_cross_lingual('text', 'prompt.wav'):
    audio = output['tts_speech']

# インストラクト（自然言語制御）
for output in cosyvoice.inference_instruct2('text', 'instruct_text', 'prompt.wav'):
    audio = output['tts_speech']

# ストリーミング
for output in cosyvoice.inference_zero_shot('text', 'prompt', 'audio.wav', stream=True):
    # チャンクごとに返却
```

## 最適化オプション

```python
# JITコンパイル
cosyvoice = AutoModel(model_dir='...', load_jit=True)

# TensorRT（Linux GPU）
cosyvoice = AutoModel(model_dir='...', load_trt=True)

# vLLM
cosyvoice = AutoModel(model_dir='...', load_vllm=True)

# FP16
cosyvoice = AutoModel(model_dir='...', fp16=True)
```

## 設定ファイル

- モデル設定: `examples/libritts/{cosyvoice,cosyvoice2,cosyvoice3}/conf/*.yaml`
- 各モデルディレクトリ: `cosyvoice.yaml`, `*.onnx`, `*.pt`, `spk2info.pt`

## 既知の注意点

1. **Matcha-TTS**: サブモジュール必須。"No module named 'matcha'"エラーは `git submodule update --init --recursive` で解決
2. **テキストフロントエンド**: ttsfrd（推奨）がない場合wetextにフォールバック
3. **vLLMバージョン**: v0.9.0とv0.11.0+で依存関係が異なる（別環境推奨）
4. **TensorRT**: Linux GPUのみ対応

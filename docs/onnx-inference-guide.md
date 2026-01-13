# CosyVoice3 ONNX推論ガイド

Pure ONNX推論（PyTorchモデル不要）の環境構築から実行までの手順書。

---

## 1. 前提条件

- Python 3.10以上
- uv（Pythonパッケージマネージャー）
- 約4GBのディスク空き容量（ONNXモデル用）

---

## 2. 環境構築

### 2.1 リポジトリのクローン

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive
```

### 2.2 uvのインストール（未インストールの場合）

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.3 Python環境の構築

```bash
# 仮想環境の作成と依存パッケージのインストール
uv sync
```

---

## 3. モデルのダウンロード

### 3.1 CosyVoice3モデルのダウンロード

```bash
uv run python -c "
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
"
```

### 3.2 ONNXファイルの確認

ダウンロード後、以下のファイルが必要：

**モデルディレクトリ直下** (`pretrained_models/Fun-CosyVoice3-0.5B/`):
```
campplus.onnx              # 28MB - 話者埋め込み抽出
speech_tokenizer_v3.onnx   # 969MB - 音声トークン化
```

**onnxサブディレクトリ** (`pretrained_models/Fun-CosyVoice3-0.5B/onnx/`):
```
text_embedding_fp32.onnx           # 544MB - テキスト埋め込み
llm_backbone_initial_fp16.onnx     # 717MB - LLM初回パス
llm_backbone_decode_fp16.onnx      # 717MB - LLMデコードステップ
llm_decoder_fp16.onnx              # 12MB  - logits出力
llm_speech_embedding_fp16.onnx     # 12MB  - 音声トークン埋め込み
flow_token_embedding_fp16.onnx     # 1MB   - Flowトークン埋め込み
flow_pre_lookahead_fp16.onnx       # 1MB   - Flow Pre-lookahead
flow_speaker_projection_fp16.onnx  # 31KB  - 話者投影
flow.decoder.estimator.fp16.onnx   # 664MB - Flow DiT
hift_f0_predictor_fp32.onnx        # 13MB  - F0予測
hift_source_generator_fp32.onnx    # 259MB - ソース生成
hift_decoder_fp32.onnx             # 70MB  - HiFTデコーダー
```

---

## 4. 推論の実行

### 4.1 基本的な使い方

```bash
uv run python scripts/onnx_inference_pure.py \
    --text "<|en|>Hello, this is a test." \
    --prompt_wav asset/prompts/en_female_nova_greeting.wav \
    --output output.wav
```

### 4.2 コマンドライン引数

| 引数 | 必須 | 説明 |
|------|------|------|
| `--text` | Yes | 合成するテキスト（言語タグ付き） |
| `--prompt_wav` | Yes | プロンプト音声ファイルのパス |
| `--output` | No | 出力ファイルパス（デフォルト: output_onnx_pure.wav） |
| `--model_dir` | No | モデルディレクトリ（デフォルト: pretrained_models/Fun-CosyVoice3-0.5B） |
| `--fp32` | No | FP32精度を使用（デフォルト: FP16） |

### 4.3 言語タグ

テキストの先頭に言語タグを付ける：

| 言語 | タグ | 例 |
|------|------|-----|
| 英語 | `<\|en\|>` | `<\|en\|>Hello world.` |
| 日本語 | `<\|ja\|>` | `<\|ja\|>こんにちは。` |
| 中国語 | `<\|zh\|>` | `<\|zh\|>你好。` |
| 韓国語 | `<\|ko\|>` | `<\|ko\|>안녕하세요.` |

---

## 5. プロンプト音声

### 5.1 重要事項

CosyVoiceは音声クローニングTTSシステムのため、**プロンプト音声は必須**です。

プロンプト音声なしでは：
- ランダムな話者特性になる
- 音声品質が低下する
- 先頭に「a~」のような異音が発生する

### 5.2 プロンプト音声の要件

| 項目 | 推奨値 |
|------|--------|
| 長さ | 3〜10秒 |
| フォーマット | WAV（他形式もlibrosa経由で対応） |
| サンプルレート | 任意（内部で変換される） |
| 品質 | 明瞭な音声、背景ノイズ最小限 |

### 5.3 テスト用プロンプト音声

`asset/prompts/` に英語のテスト用音声が用意されています：

**女性ボイス:**
- `en_female_nova_greeting.wav`
- `en_female_shimmer_greeting.wav`
- `en_female_alloy_greeting.wav`

**男性ボイス:**
- `en_male_echo_greeting.wav`
- `en_male_fable_greeting.wav`
- `en_male_onyx_greeting.wav`

各ボイスに greeting / story / technical の3種類があります。
対応するテキストは `asset/prompts/README.md` を参照してください。

---

## 6. 実行例

### 6.1 英語（女性ボイス）

```bash
uv run python scripts/onnx_inference_pure.py \
    --text "<|en|>Welcome to the future of voice synthesis." \
    --prompt_wav asset/prompts/en_female_nova_greeting.wav \
    --output english_female.wav
```

### 6.2 英語（男性ボイス）

```bash
uv run python scripts/onnx_inference_pure.py \
    --text "<|en|>This is a demonstration of voice cloning technology." \
    --prompt_wav asset/prompts/en_male_onyx_greeting.wav \
    --output english_male.wav
```

### 6.3 日本語

```bash
uv run python scripts/onnx_inference_pure.py \
    --text "<|ja|>これは音声合成のテストです。" \
    --prompt_wav asset/prompts/en_female_nova_greeting.wav \
    --output japanese.wav
```

---

## 7. 処理の流れ

```
1. プロンプト音声処理
   ├── campplus.onnx → 話者埋め込み抽出 (192次元)
   ├── speech_tokenizer_v3.onnx → 音声トークン抽出
   └── librosa → メルスペクトログラム抽出

2. LLM推論（自己回帰生成）
   ├── text_embedding → テキストトークン埋め込み
   ├── llm_speech_embedding → 音声トークン埋め込み
   ├── llm_backbone_initial → 初回パス（KVキャッシュ生成）
   ├── llm_backbone_decode → デコードステップ（繰り返し）
   └── llm_decoder → logits → トークンサンプリング

3. Flow推論（メルスペクトログラム生成）
   ├── flow_token_embedding → トークン埋め込み
   ├── flow_pre_lookahead → 特徴抽出
   ├── flow_speaker_projection → 話者埋め込み投影
   └── flow.decoder.estimator → DiT (10ステップ Euler Solver)

4. HiFT推論（波形生成）
   ├── hift_f0_predictor → F0予測
   ├── hift_source_generator → ソース信号生成
   ├── STFT → ソース信号のスペクトル分解
   ├── hift_decoder → マグニチュード・位相予測
   └── ISTFT → 波形再構成
```

---

## 8. パフォーマンス

### 8.1 テスト環境

- CPU: Intel Core（CPUのみ、GPU未使用）
- ONNX Runtime: CPUExecutionProvider

### 8.2 処理時間の目安

| フェーズ | 処理時間 |
|---------|---------|
| プロンプト処理 | 2〜3秒 |
| LLM推論 | 100〜200秒 |
| Flow推論 | 40〜100秒 |
| HiFT推論 | 2〜5秒 |

**注意**: CPUのみの処理のためRTF（Real-Time Factor）は10〜30程度。
GPUを使用する場合は大幅に高速化が期待できます。

---

## 9. トラブルシューティング

### 9.1 モデルファイルが見つからない

```
Error: Model file not found
```

**解決策**: モデルをダウンロードしてください（セクション3参照）

### 9.2 プロンプト音声が必要

```
ValueError: prompt_wav is required for CosyVoice inference.
```

**解決策**: `--prompt_wav` 引数でプロンプト音声を指定してください

### 9.3 メモリ不足

```
onnxruntime.capi.onnxruntime_pybind11_state.Fail: Failed to allocate memory
```

**解決策**:
- 他のアプリケーションを終了する
- `--fp32` オプションを外してFP16を使用する

---

## 10. ファイル構成

```
CosyVoice/
├── scripts/
│   └── onnx_inference_pure.py    # メイン推論スクリプト
├── asset/
│   └── prompts/                  # テスト用プロンプト音声
│       ├── README.md             # プロンプトテキスト対応表
│       ├── en_female_*.wav       # 女性ボイス
│       └── en_male_*.wav         # 男性ボイス
├── pretrained_models/
│   └── Fun-CosyVoice3-0.5B/
│       ├── campplus.onnx         # 話者埋め込み
│       ├── speech_tokenizer_v3.onnx  # 音声トークナイザー
│       ├── CosyVoice-BlankEN/    # Qwen2トークナイザー
│       └── onnx/                 # 推論用ONNXモデル
└── docs/
    ├── onnx-inference-guide.md   # このドキュメント
    └── onnx-export-implementation.md  # 実装詳細
```

---

## 11. 関連ドキュメント

- [ONNX Export 実装ドキュメント](onnx-export-implementation.md) - エクスポート詳細、技術仕様
- [プロンプト音声 README](../asset/prompts/README.md) - テスト用音声のテキスト対応表

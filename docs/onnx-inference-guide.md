# CosyVoice3 ONNX推論ガイド

Pure ONNX推論（PyTorchモデル不要）の環境構築から実行までの手順書。

---

## 1. 前提条件

- Python 3.10.x（3.10以上3.11未満）
- uv（Pythonパッケージマネージャー）
- 約4GBのディスク空き容量（ONNXモデル用）

---

## 2. 環境構築

ONNX推論専用の軽量環境を一から構築する手順です。PyTorchや学習用依存関係は不要です。

### 2.1 推論スクリプトの取得

推論スクリプトを取得するため、リポジトリをクローンします：

```bash
git clone https://github.com/FunAudioLLM/CosyVoice.git
```

**注意**: ONNX推論のみの場合、サブモジュール（`--recursive`）は不要です。

### 2.2 uvのインストール（未インストールの場合）

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.3 Python環境の構築（ONNX推論専用）

```bash
# 新規プロジェクトの作成（Python 3.10を指定）
uv init cosyvoice-onnx --python 3.10
cd cosyvoice-onnx

# 必要なパッケージのインストール（バージョン指定重要）
uv add "onnxruntime==1.18.0" "numpy==1.26.4" "soundfile==0.12.1" "librosa==0.10.2" "transformers==4.51.3" "scipy==1.13.1" "modelscope==1.20.0" "huggingface_hub>=0.30.0"
```

**注意**: `--python 3.10`フラグは必須です。onnxruntime 1.18.0はPython 3.8〜3.12のみサポートしています。

**重要: バージョン互換性**

オリジナルのCosyVoice（PyTorch版）も同じバージョン制約があります。これはONNX推論固有の制限ではありません。

- `onnxruntime==1.18.0`: 新しいバージョン(1.19+)はFP16モデルと互換性問題あり
- `numpy==1.26.4`: ONNX Runtime 1.18.0はNumPy 2.xと非互換（本家CosyVoiceも同じ）

**パッケージ説明:**
| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| `onnxruntime` | 1.18.0 | ONNX推論エンジン |
| `numpy` | 1.26.4 | 数値計算（1.x系必須） |
| `soundfile` | 0.12.1 | WAVファイル出力 |
| `librosa` | 0.10.2 | 音声読み込み、メルスペクトログラム抽出 |
| `transformers` | 4.51.3 | Qwen2トークナイザー |
| `scipy` | 1.13.1 | 信号処理（zoom等） |
| `modelscope` | 1.20.0 | モデルダウンロード |
| `huggingface_hub` | >=0.30.0 | Hugging Faceからのダウンロード（transformers依存） |

**GPU使用時（オプション）:**
```bash
# CUDA対応版（バージョン固定必須）
uv remove onnxruntime && uv add "onnxruntime-gpu==1.18.0"
```

**最終的なディレクトリ構成:**
```
cosyvoice-onnx/
├── pretrained_models/
│   └── Fun-CosyVoice3-0.5B/
│       ├── CosyVoice-BlankEN/      # トークナイザー
│       └── onnx/                    # ONNXモデル
│           ├── *.onnx               # モデルファイル
│           ├── scripts/             # 推論スクリプト
│           │   └── onnx_inference_pure.py
│           └── prompts/             # サンプル音声
│               └── *.wav
├── pyproject.toml
└── uv.lock
```

---

## 3. モデルのダウンロード

### 3.1 ONNXモデルのダウンロード（Hugging Face）

推論スクリプトとサンプルプロンプト音声も含まれています：

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('ayousanz/cosy-voice3-onnx', local_dir='pretrained_models/Fun-CosyVoice3-0.5B/onnx')"
```

**リポジトリ:** https://huggingface.co/ayousanz/cosy-voice3-onnx

**含まれるファイル:**
- ONNXモデル（14ファイル、約3.8GB）
- `scripts/onnx_inference_pure.py` - 推論スクリプト
- `prompts/en_female_nova_greeting.wav` - 女性サンプル音声
- `prompts/en_male_onyx_greeting.wav` - 男性サンプル音声

### 3.2 トークナイザーのダウンロード（ModelScope）

ONNXモデルに加えて、Qwen2トークナイザーが必要です：

```bash
uv run python -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B', allow_patterns=['CosyVoice-BlankEN/*.json', 'CosyVoice-BlankEN/*.txt'])"
```

### 3.3 ファイル配置の確認

ダウンロード後、以下の構成になっていることを確認：

```
pretrained_models/Fun-CosyVoice3-0.5B/
├── CosyVoice-BlankEN/           # Qwen2トークナイザー（ModelScopeから）
│   ├── tokenizer.json
│   └── ...
└── onnx/                        # ONNXモデル（Hugging Faceから）
    ├── campplus.onnx            # 28MB - 話者埋め込み抽出
    ├── speech_tokenizer_v3.onnx # 969MB - 音声トークン化
    ├── text_embedding_fp32.onnx # 544MB - テキスト埋め込み
    ├── llm_backbone_initial_fp16.onnx  # 717MB - LLM初回パス
    ├── llm_backbone_decode_fp16.onnx   # 717MB - LLMデコードステップ
    ├── llm_decoder_fp16.onnx           # 12MB  - logits出力
    ├── llm_speech_embedding_fp16.onnx  # 12MB  - 音声トークン埋め込み
    ├── flow_token_embedding_fp16.onnx  # 1MB   - Flowトークン埋め込み
    ├── flow_pre_lookahead_fp16.onnx    # 1MB   - Flow Pre-lookahead
    ├── flow_speaker_projection_fp16.onnx # 31KB - 話者投影
    ├── flow.decoder.estimator.fp16.onnx  # 664MB - Flow DiT
    ├── hift_f0_predictor_fp32.onnx     # 13MB  - F0予測
    ├── hift_source_generator_fp32.onnx # 259MB - ソース生成
    └── hift_decoder_fp32.onnx          # 70MB  - HiFTデコーダー
```

---

## 4. 推論の実行

### 4.1 基本的な使い方（Zero-Shotモード）

CosyVoice3はZero-Shot音声クローニングを使用します。プロンプト音声とその文字起こし（prompt_text）が必須です。

```bash
# Windows PowerShell（1行で実行）
uv run python pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py --text "Hello, this is a test." --prompt_wav pretrained_models/Fun-CosyVoice3-0.5B/onnx/prompts/en_female_nova_greeting.wav --prompt_text "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions." --output output.wav

# Linux/macOS
uv run python pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py \
    --text "Hello, this is a test." \
    --prompt_wav pretrained_models/Fun-CosyVoice3-0.5B/onnx/prompts/en_female_nova_greeting.wav \
    --prompt_text "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions." \
    --output output.wav
```

### 4.2 コマンドライン引数

| 引数 | 必須 | 説明 |
|------|------|------|
| `--text` | Yes | 合成するテキスト（言語タグは**不要**） |
| `--prompt_wav` | Yes | プロンプト音声ファイルのパス |
| `--prompt_text` | Yes | プロンプト音声の文字起こし |
| `--output` | No | 出力ファイルパス（デフォルト: output_onnx_pure.wav） |
| `--model_dir` | No | モデルディレクトリ（デフォルト: pretrained_models/Fun-CosyVoice3-0.5B） |
| `--fp32` | No | FP32精度を使用（デフォルト: FP16） |

### 4.3 言語について

**重要**: CosyVoice3は言語を自動検出します。`<|en|>`や`<|ja|>`などの言語タグは**使用しないでください**。これらはテキストとして発音されてしまいます。

対応言語：
- 英語、中国語、日本語、韓国語
- ドイツ語、スペイン語、フランス語、イタリア語、ロシア語
- 広東語など中国方言

---

## 5. プロンプト音声とプロンプトテキスト

### 5.1 重要事項

CosyVoice3はZero-Shot音声クローニングTTSシステムのため、**プロンプト音声**と**プロンプトテキスト**の両方が必須です。

- **プロンプト音声**: 参照となる音声サンプル
- **プロンプトテキスト**: プロンプト音声の内容の文字起こし

これにより高品質な音声クローニングが実現されます。

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

サンプルプロンプト音声のテキスト:
- `en_female_nova_greeting.wav`: "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions."
- `en_male_onyx_greeting.wav`: "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions."

### 6.1 英語（女性ボイス）

```bash
uv run python pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py \
    --text "Welcome to the future of voice synthesis." \
    --prompt_wav pretrained_models/Fun-CosyVoice3-0.5B/onnx/prompts/en_female_nova_greeting.wav \
    --prompt_text "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions." \
    --output english_female.wav
```

### 6.2 英語（男性ボイス）

```bash
uv run python pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py \
    --text "This is a demonstration of voice cloning technology." \
    --prompt_wav pretrained_models/Fun-CosyVoice3-0.5B/onnx/prompts/en_male_onyx_greeting.wav \
    --prompt_text "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions." \
    --output english_male.wav
```

### 6.3 日本語

```bash
uv run python pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py \
    --text "これは音声合成のテストです。" \
    --prompt_wav pretrained_models/Fun-CosyVoice3-0.5B/onnx/prompts/en_female_nova_greeting.wav \
    --prompt_text "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions." \
    --output japanese.wav
```

---

## 7. 処理の流れ

```
1. プロンプト音声処理
   ├── campplus.onnx → 話者埋め込み抽出 (192次元)
   ├── speech_tokenizer_v3.onnx → 音声トークン抽出（LLMコンテキスト用）
   └── librosa → メルスペクトログラム抽出（Flow条件付け用）

2. LLM推論（Zero-Shotモード）
   ├── text_embedding → [prompt_text + tts_text] 埋め込み
   ├── llm_speech_embedding → プロンプト音声トークン埋め込み
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
   ├── STFT (numpy) → ソース信号のスペクトル分解
   ├── hift_decoder → マグニチュード・位相予測
   └── ISTFT (numpy) → 波形再構成
```

### 7.1 完全なPyTorchフリー実装

この推論スクリプトはPyTorch依存なしで動作します。すべての処理はONNX RuntimeとNumPy/SciPyで実装されています。

**HiFT STFT/ISTFT パラメータ（CosyVoice3固有）:**

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `upsample_rates` | [8, 5, 3] | HiFTアップサンプリング倍率（合計120倍） |
| `n_fft` | 16 | FFTウィンドウサイズ |
| `hop_length` | 4 | ホップ長 |
| `center` | True | 信号のパディング（PyTorchと同じ） |

**注意**: CosyVoice2ではupsample_rates=[8, 8]（64倍）ですが、CosyVoice3では[8, 5, 3]（120倍）です。

期待されるSTFTフレーム数 = mel_frames × 120 + 1

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

### 9.2 プロンプト音声・テキストが必要

```
ValueError: prompt_wav is required for CosyVoice inference.
ValueError: prompt_text is required for zero-shot mode.
```

**解決策**: `--prompt_wav` と `--prompt_text` の両方を指定してください

### 9.3 言語タグが発音される

`<|en|>`や`<|ja|>`が音声として発音される場合。

**解決策**: 言語タグをテキストから削除してください。CosyVoice3は言語を自動検出します。

```bash
# 間違い
--text "<|en|>Hello world"

# 正しい
--text "Hello world"
```

### 9.4 メモリ不足

```
onnxruntime.capi.onnxruntime_pybind11_state.Fail: Failed to allocate memory
```

**解決策**:
- 他のアプリケーションを終了する
- `--fp32` オプションを外してFP16を使用する

### 9.5 ONNX Runtimeバージョンエラー（FP16モデル）

```
RuntimeException: Attempting to get index by a name which does not exist
```

**解決策**: `onnxruntime==1.18.0` を使用してください。1.19以降はFP16モデルと互換性問題があります。

### 9.5 NumPy 2.x非互換エラー

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**解決策**: `numpy==1.26.4` を使用してください。これはONNX推論固有の制限ではなく、オリジナルのCosyVoice（PyTorch版）も同じ制約があります。

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

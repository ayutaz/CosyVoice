# はじめに

CosyVoice3は、Alibaba FunAudioLLMが開発したLLMベースの音声合成（TTS）システムです。9言語以上に対応し、ゼロショット音声クローニングが可能な最新のTTSモデルです。

今回、このCosyVoice3を完全にONNX化し、PyTorchなしで推論できるようにしました。Unity Sentisでの利用やエッジデバイスへの展開を見据えた取り組みです。

本記事では、ONNXエクスポートから推論実装までの技術的な詳細を解説します。

## 成果物

- **ONNXモデル**: [Hugging Face - ayousanz/cosy-voice3-onnx](https://huggingface.co/ayousanz/cosy-voice3-onnx)
- **推論スクリプト**: 上記リポジトリに同梱

# CosyVoice3のアーキテクチャ

CosyVoice3は4段階のパイプラインで構成されています。

```
テキスト入力
    ↓
[1. Tokenizer] テキスト → トークンID
    ↓
[2. LLM] Qwen2ベース → 音声トークン生成（自己回帰）
    ↓
[3. Flow] DiT + Euler Solver → メルスペクトログラム
    ↓
[4. HiFT] F0予測 + Source生成 + Decoder → 24kHz音声波形
```

## Zero-Shot音声クローニングの仕組み

Zero-Shotモードでは、プロンプト音声から話者特徴を抽出し、任意のテキストをその声で読み上げます。

```
プロンプト音声
    ├── [CAMPPlus] → 話者埋め込み（192次元）
    ├── [Speech Tokenizer] → 音声トークン（LLMコンテキスト用）
    └── [librosa] → メルスペクトログラム（Flow条件付け用）
```

# 開発環境

| 項目 | バージョン |
|------|-----------|
| OS | Windows 11 |
| Python | 3.10 |
| PyTorch | 2.5.1+cu124（エクスポート時のみ） |
| ONNX Runtime | 1.18.0 |
| NumPy | 1.26.4 |

# ONNXエクスポート

## エクスポート戦略

CosyVoice3を14個のONNXモデルに分割してエクスポートしました。分割の理由は以下の通りです。

1. **LLMのKVキャッシュ**: 初回パスとデコードパスで入出力形状が異なる
2. **精度要件**: HiFTはFP32必須、LLM/FlowはFP16で動作可能
3. **Unity Sentis対応**: 動的形状の制約を回避

## 生成されたONNXファイル

| ファイル | サイズ | 精度 | 用途 |
|---------|-------|------|------|
| text_embedding_fp32.onnx | 544MB | FP32 | テキスト埋め込み（Qwen2） |
| llm_backbone_initial_fp16.onnx | 717MB | FP16 | LLM初回パス |
| llm_backbone_decode_fp16.onnx | 717MB | FP16 | LLMデコードステップ |
| llm_decoder_fp16.onnx | 12MB | FP16 | Logits出力 |
| llm_speech_embedding_fp16.onnx | 12MB | FP16 | 音声トークン埋め込み |
| flow_token_embedding_fp16.onnx | 1MB | FP16 | Flowトークン埋め込み |
| flow_pre_lookahead_fp16.onnx | 1MB | FP16 | Flow前処理 |
| flow_speaker_projection_fp16.onnx | 31KB | FP16 | 話者投影 |
| flow.decoder.estimator.fp16.onnx | 664MB | FP16 | Flow DiT |
| hift_f0_predictor_fp32.onnx | 13MB | FP32 | F0予測 |
| hift_source_generator_fp32.onnx | 259MB | FP32 | Source信号生成 |
| hift_decoder_fp32.onnx | 70MB | FP32 | HiFTデコーダー |
| campplus.onnx | 28MB | FP32 | 話者埋め込み |
| speech_tokenizer_v3.onnx | 969MB | FP32 | 音声トークナイザー |

合計サイズは約3.8GBです。

## LLMのKVキャッシュ分割

LLMの自己回帰推論では、KVキャッシュの効率的な管理が重要です。CosyVoice3のLLM（Qwen2ベース）は24層あり、各層にKey/Valueキャッシュが必要です。

**初回パス（llm_backbone_initial）**:
- 入力: 全コンテキスト（プロンプト + テキスト埋め込み）
- 出力: hidden_states + 24層分のKVキャッシュ

**デコードパス（llm_backbone_decode）**:
- 入力: 1トークンの埋め込み + 前ステップのKVキャッシュ
- 出力: hidden_states + 更新されたKVキャッシュ

```python
# エクスポート時の入出力定義
dynamic_axes = {
    'input_embeds': {0: 'batch', 1: 'seq_len'},
    'attention_mask': {0: 'batch', 1: 'total_len'},
}
for i in range(24):
    dynamic_axes[f'past_key_{i}'] = {0: 'batch', 2: 'past_len'}
    dynamic_axes[f'past_value_{i}'] = {0: 'batch', 2: 'past_len'}
```

## HiFTのFP32必須問題

当初、HiFTコンポーネントもFP16でエクスポートしましたが、生成音声にノイズが入る問題が発生しました。

調査の結果、HiFTのSTFT/ISTFT処理で数値精度が重要であることがわかりました。FP32に変更することで問題が解決しました。

```python
# HiFTはFP32でエクスポート
torch.onnx.export(
    hift_decoder,
    dummy_input,
    "hift_decoder_fp32.onnx",
    # FP16変換なし
)
```

# PyTorchフリー推論の実装

## 依存パッケージ

PyTorchなしで推論するため、以下の最小構成で環境を構築しました。

```bash
uv init cosyvoice-onnx --python 3.10
cd cosyvoice-onnx
uv add "onnxruntime==1.18.0" "numpy==1.26.4" "soundfile==0.12.1" \
       "librosa==0.10.2" "transformers==4.51.3" "scipy==1.13.1"
```

**バージョン固定の理由**:
- `onnxruntime==1.18.0`: 1.19以降はFP16モデルで互換性問題あり
- `numpy==1.26.4`: ONNX Runtime 1.18.0はNumPy 2.x非対応

## NumPy/SciPyによるSTFT/ISTFT実装

HiFTの波形生成にはSTFT/ISTFTが必要ですが、PyTorchのtorch.stftを使えないため、NumPy/SciPyで実装しました。

CosyVoice3のHiFTは特殊なパラメータを使用しています。

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| n_fft | 16 | 非常に小さい値 |
| hop_length | 4 | |
| upsample_rates | [8, 5, 3] | 120倍アップサンプル |

```python
def stft_numpy(x, n_fft=16, hop_length=4, center=True):
    """NumPyによるSTFT実装"""
    if center:
        pad_amount = n_fft // 2
        x = np.pad(x, pad_amount, mode='reflect')

    # ハニング窓
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)

    # フレーム分割
    num_frames = (len(x) - n_fft) // hop_length + 1
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(num_frames, n_fft),
        strides=(x.strides[0] * hop_length, x.strides[0])
    ).copy()

    # 窓関数適用してFFT
    windowed = frames * window
    return np.fft.rfft(windowed, axis=1).astype(np.complex64)

def istft_numpy(stft_matrix, hop_length=4, n_fft=16, length=None, center=True):
    """NumPyによるISTFT実装"""
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)

    # IFFT
    time_frames = np.fft.irfft(stft_matrix, n=n_fft, axis=1).real.astype(np.float32)

    # Overlap-Add
    num_frames = time_frames.shape[0]
    expected_length = n_fft + hop_length * (num_frames - 1)
    output = np.zeros(expected_length, dtype=np.float32)
    window_sum = np.zeros(expected_length, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_length
        output[start:start + n_fft] += time_frames[i] * window
        window_sum[start:start + n_fft] += window ** 2

    # 正規化
    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    # centerパディングの除去
    if center:
        pad = n_fft // 2
        output = output[pad:-pad]

    if length is not None:
        output = output[:length]

    return output
```

## 推論ループの実装

Zero-Shotモードの推論フローは以下の通りです。

```python
class CosyVoiceONNXInference:
    def inference_zero_shot(self, text, prompt_text, prompt_wav):
        # 1. プロンプト音声の処理
        speech_feat = self.extract_mel(prompt_wav)  # メル抽出
        speaker_embedding = self.campplus(prompt_wav)  # 話者埋め込み
        prompt_speech_tokens = self.speech_tokenizer(prompt_wav)  # 音声トークン

        # 2. テキストトークン化
        prompt_ids = self.tokenizer.encode(prompt_text)
        tts_ids = self.tokenizer.encode(text)

        # 3. 埋め込み生成
        text_embeds = self.text_embedding(prompt_ids + tts_ids)
        speech_embeds = self.speech_embedding(prompt_speech_tokens)

        # 4. LLM推論（自己回帰）
        # 初回パス
        hidden, kv_cache = self.llm_initial(text_embeds, speech_embeds)

        # デコードループ
        speech_tokens = []
        for _ in range(max_length):
            logits = self.llm_decoder(hidden)
            token = sample_token(logits)
            if token == eos_token:
                break
            speech_tokens.append(token)

            token_embed = self.speech_embedding([token])
            hidden, kv_cache = self.llm_decode(token_embed, kv_cache)

        # 5. Flow推論（メル生成）
        mel = self.flow_inference(speech_tokens, speaker_embedding, speech_feat)

        # 6. HiFT推論（波形生成）
        audio = self.hift_inference(mel)

        return audio
```

# 発見した問題と解決策

## 言語タグが発音される問題

当初、CosyVoice3の多言語対応のため `<|en|>` や `<|ja|>` などの言語タグを使用していました。しかし、生成音声でこれらのタグが文字通り発音されてしまう問題が発生しました。

**原因**: 言語タグはQwen2トークナイザーの特殊トークンではなく、通常のテキストとしてトークン化されていました。

```python
# 言語タグのトークン化結果
tokenizer.encode("<|en|>")
# → [27, 91, 268, 91, 29]  # '<', '|', 'en', '|', '>' の5トークン
```

**解決策**: 言語タグを使用せず、CosyVoice3の自動言語検出に任せることで解決しました。

```python
# NG: 言語タグあり
text = "<|en|>Hello, this is a test."

# OK: 言語タグなし（自動検出）
text = "Hello, this is a test."
```

## ONNX Runtimeのバージョン問題

ONNX Runtime 1.19以降でFP16モデルを読み込むと以下のエラーが発生しました。

```
RuntimeException: Attempting to get index by a name which does not exist
```

これはONNX Runtime 1.19以降のFP16処理の変更に起因する問題です。ONNX Runtime 1.18.0を使用することで回避しました。

# 使い方

## 環境構築

```bash
# 新規プロジェクト作成
uv init cosyvoice-onnx --python 3.10
cd cosyvoice-onnx

# パッケージインストール
uv add "onnxruntime==1.18.0" "numpy==1.26.4" "soundfile==0.12.1" \
       "librosa==0.10.2" "transformers==4.51.3" "scipy==1.13.1" \
       "huggingface_hub>=0.30.0"
```

## モデルダウンロード

```bash
# ONNXモデル + トークナイザー + 推論スクリプト + サンプル音声（すべて含まれています）
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('ayousanz/cosy-voice3-onnx',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B/onnx')
"
```

トークナイザーファイル（`vocab.json`, `merges.txt`, `tokenizer_config.json`）もリポジトリに含まれているため、追加のダウンロードは不要です。

## 推論実行

```bash
# 英語
uv run python pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py \
    --text "Hello, this is a test of ONNX inference." \
    --prompt_wav pretrained_models/Fun-CosyVoice3-0.5B/onnx/prompts/en_female_nova_greeting.wav \
    --prompt_text "Hello, my name is Sarah." \
    --output output.wav

# 日本語
uv run python pretrained_models/Fun-CosyVoice3-0.5B/onnx/scripts/onnx_inference_pure.py \
    --text "こんにちは、今日はいい天気ですね。" \
    --prompt_wav pretrained_models/Fun-CosyVoice3-0.5B/onnx/prompts/en_female_nova_greeting.wav \
    --prompt_text "Hello, my name is Sarah." \
    --output output_ja.wav
```

# まとめ

CosyVoice3を14個のONNXモデルに分割し、PyTorchなしで推論できるようになりました。

**技術的なポイント**:
- LLMのKVキャッシュを初回パスとデコードパスに分離
- HiFTはFP32必須（精度問題）
- STFT/ISTFTをNumPy/SciPyで実装
- 言語タグは使用せず自動検出に任せる

今後はUnity Sentisへの移植を予定しています。

# 参考リンク

- [CosyVoice公式リポジトリ](https://github.com/FunAudioLLM/CosyVoice)
- [元モデル（ModelScope）](https://modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [CosyVoice論文](https://arxiv.org/abs/2407.05407)
- [ONNXモデル（Hugging Face）](https://huggingface.co/ayousanz/cosy-voice3-onnx)

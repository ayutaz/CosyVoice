# CosyVoice3 Unity完結ONNX戦略

## 概要

**目標**: CosyVoice3の全コンポーネントをONNXにエクスポートし、Unity Sentis内で完結した推論を実現する。

**制約**: サーバー不要、完全オフライン動作

---

## 1. CosyVoice3 アーキテクチャ分析

### 1.1 推論パイプライン

```
テキスト入力
    ↓
[Tokenizer] テキスト → トークンID (C#実装)
    ↓
[LLM - CosyVoice3LM] トークン → 音声トークン (Qwen2ForCausalLM)
    ↓
[Flow - CausalMaskedDiffWithDiT] 音声トークン → メルスペクトログラム
    ↓
[HiFT - HiFTGenerator] メルスペクトログラム → 波形 (24kHz)
```

### 1.2 コンポーネント詳細

| コンポーネント | クラス | 主要モジュール | FP32サイズ | FP16サイズ |
|--------------|--------|--------------|----------|----------|
| LLM | `CosyVoice3LM` | `Qwen2ForCausalLM` | ~400MB | ~200MB |
| Flow | `CausalMaskedDiffWithDiT` | `decoder.estimator` | ~80MB | ~40MB |
| HiFT | `HiFTGenerator` | `f0_predictor`, `decode` | ~20MB | ~20MB (FP32推奨) |
| Speech Tokenizer | `campplus.onnx` | 話者埋め込み | ~10MB | ~5MB |

**合計推定サイズ: FP32で~510MB、FP16で~265MB**

> ⚠️ **重要**: INT8量子化はTTSで音質劣化（背景ノイズ等）のリスクがあるため非推奨。
> 詳細は `docs/quantization-analysis.md` を参照。

---

## 2. ONNXエクスポート戦略

### 2.1 LLM (CosyVoice3LM)

**構成:**
- `Qwen2ForCausalLM`: HuggingFace Transformersモデル
- `speech_embedding`: nn.Embedding (speech_token_size + 200, llm_input_size)
- `llm_decoder`: nn.Linear (llm_output_size, speech_token_size + 200)

**エクスポート方法:**

#### 方法A: HuggingFace Optimum使用

```bash
# Qwen2ベースモデルのエクスポート
optimum-cli export onnx \
    --model Qwen/Qwen2-0.5B \
    --task text-generation-with-past \
    --opset 15 \
    output_dir/qwen2_onnx/
```

**出力ファイル:**
- `model.onnx` または `decoder_model.onnx`
- `decoder_with_past_model.onnx` (KVキャッシュ付き)

#### 方法B: カスタムONNXエクスポート

```python
import torch

# CosyVoice3LMのラッパー
class LLMWrapper(torch.nn.Module):
    def __init__(self, llm_model):
        super().__init__()
        self.qwen = llm_model.llm.model
        self.speech_embedding = llm_model.speech_embedding
        self.llm_decoder = llm_model.llm_decoder

    def forward(self, inputs_embeds, attention_mask, past_key_values=None):
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states[-1]
        logits = self.llm_decoder(hidden_states)
        return logits, outputs.past_key_values

# エクスポート
torch.onnx.export(
    LLMWrapper(cosyvoice_model.llm),
    (dummy_inputs_embeds, dummy_attention_mask),
    "llm_qwen2.onnx",
    input_names=["inputs_embeds", "attention_mask"],
    output_names=["logits", "past_key_values"],
    dynamic_axes={
        "inputs_embeds": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"}
    },
    opset_version=15
)
```

**KVキャッシュの扱い:**
- 推論時はC#側でKVキャッシュを管理
- 各ステップで前のKVを入力として渡す
- [Esperanto Technologies方式](https://www.esperanto.ai/blog/exporting-slms-to-onnx-with-kv-cache-support/)参照

### 2.2 Flow Decoder Estimator

**現状:**
- 公式スクリプト `cosyvoice/bin/export_onnx.py` で対応済み

**使用方法:**
```bash
python cosyvoice/bin/export_onnx.py \
    --model_dir pretrained_models/Fun-CosyVoice3-0.5B
```

**出力:**
```
pretrained_models/Fun-CosyVoice3-0.5B/flow.decoder.estimator.fp32.onnx
```

**入出力仕様:**
```
入力:
  - x: [B, 80, T] float32 - 現在状態
  - mask: [B, 1, T] float32 - マスク
  - mu: [B, 80, T] float32 - 条件
  - t: [B] float32 - タイムステップ
  - spks: [B, 80] float32 - 話者埋め込み
  - cond: [B, 80, T] float32 - 追加条件

出力:
  - estimator_out: [B, 80, T] float32 - 推定結果
```

### 2.3 HiFT Generator

**課題:**
- `torch.stft` / `torch.istft` はONNXで複雑
- ISTFTパラメータ: n_fft=16, hop_len=4（非常に小さい）

**解決策:**

#### 方法A: STFT/ISTFT分離エクスポート

```python
class HiFTWithoutISTFT(torch.nn.Module):
    """ISTFT前までをエクスポート"""
    def __init__(self, hift):
        super().__init__()
        self.hift = hift

    def forward(self, speech_feat, source_stft):
        # decode部分（ISTFT前まで）
        x = self.hift.conv_pre(speech_feat)
        # ... 中間処理 ...
        magnitude = torch.exp(x[:, :self.hift.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.hift.istft_params["n_fft"] // 2 + 1:, :])
        return magnitude, phase

# C#側でISTFTを実装（n_fft=16なので簡単）
```

#### 方法B: 完全分離

1. **F0 Predictor** → ONNX
2. **Source Module (SineGen)** → C#実装（正弦波生成）
3. **Decode Network** → ONNX（STFT/ISTFT除く）
4. **ISTFT** → C#実装（NWaves使用）

**ISTFT C#実装 (n_fft=16の場合):**
```csharp
public class MiniISTFT
{
    private const int N_FFT = 16;
    private const int HOP_LEN = 4;
    private float[] _window;

    public MiniISTFT()
    {
        // Hann窓 (16点)
        _window = new float[N_FFT];
        for (int i = 0; i < N_FFT; i++)
            _window[i] = 0.5f * (1 - MathF.Cos(2 * MathF.PI * i / (N_FFT - 1)));
    }

    public float[] Process(float[] magnitude, float[] phase, int numFrames)
    {
        int outputLen = (numFrames - 1) * HOP_LEN + N_FFT;
        float[] output = new float[outputLen];
        float[] windowSum = new float[outputLen];

        // 逆FFT用バッファ (n_fft=16なので小さい)
        float[] realSpectrum = new float[N_FFT];
        float[] imagSpectrum = new float[N_FFT];

        for (int frame = 0; frame < numFrames; frame++)
        {
            // 複素スペクトル構築
            int numBins = N_FFT / 2 + 1; // 9ビン
            for (int bin = 0; bin < numBins; bin++)
            {
                int idx = bin * numFrames + frame;
                float mag = magnitude[idx];
                float ph = phase[idx];
                realSpectrum[bin] = mag * MathF.Cos(ph);
                imagSpectrum[bin] = mag * MathF.Sin(ph);

                // 共役対称性
                if (bin > 0 && bin < numBins - 1)
                {
                    realSpectrum[N_FFT - bin] = realSpectrum[bin];
                    imagSpectrum[N_FFT - bin] = -imagSpectrum[bin];
                }
            }

            // 16点IFFT (高速化可能)
            float[] timeSignal = IFFT16(realSpectrum, imagSpectrum);

            // オーバーラップ加算
            int frameStart = frame * HOP_LEN;
            for (int i = 0; i < N_FFT; i++)
            {
                int outIdx = frameStart + i;
                if (outIdx < outputLen)
                {
                    output[outIdx] += timeSignal[i] * _window[i] / N_FFT;
                    windowSum[outIdx] += _window[i] * _window[i];
                }
            }
        }

        // COLA正規化
        for (int i = 0; i < outputLen; i++)
            if (windowSum[i] > 1e-8f) output[i] /= windowSum[i];

        return output;
    }

    private float[] IFFT16(float[] real, float[] imag)
    {
        // 16点IFFTは直接計算またはバタフライで高速化
        // 省略...
    }
}
```

### 2.4 Speech Tokenizer (CAMPPlus)

**現状:**
- `campplus.onnx` として公開済み
- 話者埋め込み抽出用

**仕様:**
```
入力: audio [B, T] - 16kHz音声
出力: embedding [B, 192] - 話者埋め込み
```

---

## 3. Unity統合アーキテクチャ

### 3.1 ONNXモデル構成

```
Assets/CosyVoice/Models/
├── llm_qwen2.onnx              # LLM (テキスト→音声トークン)
├── llm_speech_embedding.onnx   # 音声トークン埋め込み
├── flow_estimator.onnx         # Flow Matching Decoder
├── hift_f0_predictor.onnx      # F0予測
├── hift_decoder.onnx           # HiFT Decoder (STFT/ISTFT除く)
└── campplus.onnx               # 話者埋め込み
```

### 3.2 C#実装コンポーネント

```
Assets/CosyVoice/Runtime/
├── Core/
│   └── CosyVoiceManager.cs         # メインAPI
├── Inference/
│   ├── LLMInference.cs             # LLM推論 + KVキャッシュ管理
│   ├── FlowDecoder.cs              # Flow Matching + Euler Solver
│   ├── HiFTVocoder.cs              # HiFT推論
│   └── SpeakerEncoder.cs           # CAMPPlus推論
├── Tokenizer/
│   └── CosyVoiceTokenizer.cs       # テキスト→トークンID
├── Audio/
│   ├── MiniISTFT.cs                # 16点ISTFT
│   ├── SineGenerator.cs            # 正弦波生成
│   └── AudioClipBuilder.cs         # AudioClip生成
└── Utils/
    ├── EulerSolver.cs              # ODE統合
    └── KVCacheManager.cs           # KVキャッシュ管理
```

### 3.3 推論フロー

```csharp
public class CosyVoiceManager : MonoBehaviour
{
    private LLMInference _llm;
    private FlowDecoder _flow;
    private HiFTVocoder _hift;
    private SpeakerEncoder _speaker;

    public async UniTask<AudioClip> SynthesizeAsync(
        string text,
        AudioClip promptAudio,
        CancellationToken ct = default)
    {
        // 1. テキストトークン化
        var textTokens = _tokenizer.Encode(text);

        // 2. プロンプト音声から話者埋め込み抽出
        var embedding = await _speaker.ExtractAsync(promptAudio, ct);

        // 3. LLM推論（自己回帰、KVキャッシュ使用）
        var speechTokens = await _llm.GenerateAsync(
            textTokens, embedding, ct);

        // 4. Flow Matching (Euler Solver)
        var mel = await _flow.GenerateAsync(
            speechTokens, embedding, numSteps: 10, ct);

        // 5. HiFT Vocoder
        var audio = await _hift.GenerateAsync(mel, ct);

        // 6. AudioClip生成
        return AudioClipBuilder.Build(audio, 24000, "TTS");
    }
}
```

---

## 4. 技術的課題と解決策

### 4.1 LLMサイズ問題

| 問題 | 解決策 |
|-----|-------|
| モデルサイズ (~400MB) | INT8量子化で~100MBに削減 |
| メモリ使用量 | ストリーミング推論、チャンク処理 |
| 推論速度 | KVキャッシュ必須、GPUPixelバックエンド |

**INT8量子化:**
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="llm_qwen2.onnx",
    model_output="llm_qwen2_int8.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"]
)
```

### 4.2 KVキャッシュ管理

**C#実装パターン:**
```csharp
public class KVCacheManager
{
    private Dictionary<int, Tensor<float>> _keyCache;
    private Dictionary<int, Tensor<float>> _valueCache;
    private int _maxSeqLen;

    public void Update(int layerId, Tensor<float> newK, Tensor<float> newV)
    {
        // 既存キャッシュと連結
        var existingK = _keyCache.GetValueOrDefault(layerId);
        var existingV = _valueCache.GetValueOrDefault(layerId);

        _keyCache[layerId] = Concatenate(existingK, newK, axis: 2);
        _valueCache[layerId] = Concatenate(existingV, newV, axis: 2);

        // 最大長超過時は古いものを削除
        TruncateIfNeeded(layerId);
    }
}
```

### 4.3 Unity Sentis制約

| 制約 | 対応 |
|-----|------|
| Opset 7-15 | Opset 15でエクスポート |
| FFT/IFFT未サポート | C#でMiniISTFT実装 |
| スカラーランク0 | TensorShape()使用 |
| 動的形状 | dynamic_axes指定 |

---

## 5. 実装ロードマップ

### Phase 1: 基盤構築（1-2週間）

1. ONNXエクスポートスクリプト作成
   - LLM (Qwen2) エクスポート
   - HiFT分離エクスポート
2. 基本C#ラッパー実装
   - Tensor操作ユーティリティ
   - AudioClipBuilder

### Phase 2: コア推論実装（2-3週間）

1. FlowDecoder + EulerSolver
2. HiFTVocoder + MiniISTFT
3. SpeakerEncoder (CAMPPlus)

### Phase 3: LLM統合（2-3週間）

1. LLM推論実装
2. KVキャッシュ管理
3. 自己回帰ループ

### Phase 4: 最適化（1-2週間）

1. INT8量子化適用
2. メモリ最適化
3. バッチ処理

---

## 6. 参考リソース

### ONNXエクスポート
- [HuggingFace Optimum ONNX Export](https://huggingface.co/docs/optimum-onnx/en/onnx/usage_guides/export_a_model)
- [Esperanto: SLM ONNX with KV Cache](https://www.esperanto.ai/blog/exporting-slms-to-onnx-with-kv-cache-support/)

### Unity Sentis
- [Unity Sentis Documentation](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.4/manual/index.html)
- [Sentis Supported Models](https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/supported-models.html)

### 参考実装
- uZipVoice: `C:\Users\yuta\Desktop\Private\uZipVoice`
- ZipVoice: `C:\Users\yuta\Desktop\Private\ZipVoice`
- uPiper: `C:\Users\yuta\Desktop\Private\uPiper`

---

## 7. リスクと緩和策

| リスク | 影響 | 緩和策 |
|-------|------|--------|
| LLMサイズ過大 | メモリ不足 | 量子化、モデル蒸留 |
| 推論速度低下 | リアルタイム性喪失 | GPU使用、チャンク処理 |
| ONNX互換性問題 | エクスポート失敗 | カスタムオペレーター実装 |
| 音質劣化 | ユーザー体験低下 | 量子化閾値調整 |

---

## 8. 次のアクション

1. **即座に実行**:
   - LLM (Qwen2) のONNXエクスポート検証
   - HiFTの分離エクスポートスクリプト作成

2. **検証項目**:
   - 各ONNXモデルのUnity Sentis互換性
   - メモリ使用量測定
   - 推論時間計測

3. **プロトタイプ**:
   - HiFT + Flow のみでUnity動作確認
   - 音声トークンをダミーデータで検証

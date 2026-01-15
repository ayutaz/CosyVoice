# CosyVoice3 Unity Sentis 実装ガイド

**作成日**: 2026-01-14
**ブランチ**: feature/unity-sentis-onnx

---

## 1. 概要

CosyVoice3をUnity Sentis（AI Inference Engine）で動作させるための実装ガイドです。

### 1.1 必要なONNXモデル

| モデル | サイズ | 用途 | Sentis対応 |
|--------|-------|------|-----------|
| text_embedding | 544MB | テキスト埋め込み（Qwen2） | 要再エクスポート（opset 17→15） |
| llm_backbone_initial_fp16 | 717MB | LLM初回パス | OK（opset 15） |
| llm_backbone_decode_fp16 | 717MB | LLMデコードステップ | OK（opset 15） |
| llm_decoder_fp16 | 12MB | Logits出力 | OK（opset 15） |
| llm_speech_embedding_fp16 | 12MB | 音声トークン埋め込み | OK（opset 15） |
| flow_token_embedding_fp16 | 1MB | Flowトークン埋め込み | OK（opset 15） |
| flow_pre_lookahead_fp16 | 1MB | Flow前処理 | OK（opset 15） |
| flow_speaker_projection_fp16 | 31KB | 話者投影 | OK（opset 15） |
| flow.decoder.estimator.fp16 | 664MB | Flow DiT | 要再エクスポート（opset 18→15） |
| hift_f0_predictor_fp32 | 13MB | F0予測 | OK（opset 15） |
| hift_source_generator_fp32 | 259MB | Source信号生成 | 要再エクスポート（opset 17→15） |
| hift_decoder_fp32 | 70MB | HiFTデコーダー | OK（opset 15） |
| campplus | 28MB | 話者埋め込み | 未確認 |
| speech_tokenizer_v3 | 969MB | 音声トークナイザー | 未確認 |

**合計サイズ**: 約3.8GB

---

## 2. Unity Sentis制約

### 2.1 Opsetバージョン

```
サポート範囲: opset 7-15
推奨: opset 15
```

**対応が必要なモデル**:
- flow.decoder.estimator.fp16.onnx (opset 18)
- hift_source_generator_fp32.onnx (opset 17)
- text_embedding_fp32.onnx (opset 17)

### 2.2 非サポートオペレーター

```
If                 - 条件分岐（静的グラフに変換が必要）
Log1p              - log(1+x)で代替
FFT, IFFT, RFFT, IRFFT - 信号処理系（C#で実装）
GRU, RNN           - リカレント系
Loop               - 制御フロー
QuantizeLinear, DequantizeLinear - 量子化
```

### 2.3 テンソル次元制限

```
最大次元数: 8
```

### 2.4 サポートされているオペレーター（使用中）

```
# 基本演算
Add, Sub, Mul, Div, Pow, Abs, Neg, Sqrt, Exp, Sin, Cos, Tanh, Sigmoid

# 活性化関数
Softmax, Elu, LeakyRelu, Softplus

# 畳み込み
Conv

# 正規化
LayerNormalization (サポート)

# テンソル操作
Reshape, Transpose, Concat, Split, Gather, Slice, Unsqueeze, Squeeze, Pad, Tile, Expand

# 行列演算
MatMul, Gemm

# 削減演算
ReduceMean

# 特殊オペレーター
Einsum       - サポート（CPUは全機能、GPUは1-2入力のみ）
ScatterND    - サポート
CumSum       - サポート
Reciprocal   - サポート
Resize       - サポート
```

---

## 3. Sentis用モデル再エクスポート

### 3.1 エクスポートスクリプト

```bash
# Python環境（CosyVoiceリポジトリ）
cd C:\Users\yuta\Desktop\Private\CosyVoice

# 全モデルをopset 15で再エクスポート
python scripts/export_onnx_sentis.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B

# 個別エクスポート
python scripts/export_onnx_sentis.py --export_flow    # flow.decoder.estimator
python scripts/export_onnx_sentis.py --export_text    # text_embedding
python scripts/export_onnx_sentis.py --export_source  # hift_source_generator

# 既存モデルの検証のみ
python scripts/export_onnx_sentis.py --verify_only
```

### 3.2 出力ファイル

```
pretrained_models/Fun-CosyVoice3-0.5B/onnx_sentis/
├── flow.decoder.estimator.sentis.onnx   # opset 15
├── text_embedding.sentis.onnx           # opset 15
└── hift_source_generator.sentis.onnx    # opset 15
```

---

## 4. Unity実装アーキテクチャ

### 4.1 推論パイプライン

```
テキスト入力
    ↓
[C# Tokenizer] テキスト → トークンID
    ↓
[text_embedding] トークンID → 埋め込みベクトル
    ↓
[LLM推論] 自己回帰で音声トークン生成
    ├── llm_backbone_initial (初回パス、KVキャッシュ生成)
    ├── llm_backbone_decode (デコードループ)
    ├── llm_decoder (logits出力)
    └── llm_speech_embedding (音声トークン埋め込み)
    ↓
[Flow推論] 音声トークン → メルスペクトログラム
    ├── flow_token_embedding
    ├── flow_pre_lookahead
    ├── flow_speaker_projection
    └── flow.decoder.estimator (DiT、10ステップEuler)
    ↓
[HiFT推論] メル → 波形
    ├── hift_f0_predictor
    ├── hift_source_generator
    ├── [C# STFT]
    ├── hift_decoder
    └── [C# ISTFT]
    ↓
24kHz音声波形
```

### 4.2 C#実装が必要なコンポーネント

| コンポーネント | 説明 | 難易度 |
|--------------|------|--------|
| **Qwen2 Tokenizer** | テキスト→トークンID変換 | 高 |
| **KV Cache Manager** | LLMのKey/Valueキャッシュ管理（24層） | 高 |
| **Euler Solver** | Flow ODEソルバー（10ステップ） | 中 |
| **ISTFT** | 逆短時間フーリエ変換（n_fft=16） | 低 |
| **STFT** | 短時間フーリエ変換（n_fft=16） | 低 |

---

## 5. C#実装詳細

### 5.1 ISTFT実装（n_fft=16）

CosyVoice3のHiFTは非常に小さいFFTサイズを使用しています。

```csharp
public class MiniISTFT
{
    private const int N_FFT = 16;
    private const int HOP_LENGTH = 4;
    private float[] _window;

    public MiniISTFT()
    {
        // Hann窓
        _window = new float[N_FFT];
        for (int i = 0; i < N_FFT; i++)
            _window[i] = 0.5f * (1 - MathF.Cos(2 * MathF.PI * i / (N_FFT - 1)));
    }

    public float[] Process(float[] magnitude, float[] phase, int numFrames)
    {
        int outputLen = N_FFT + HOP_LENGTH * (numFrames - 1);
        float[] output = new float[outputLen];
        float[] windowSum = new float[outputLen];

        for (int frame = 0; frame < numFrames; frame++)
        {
            // 複素スペクトル構築 (9ビン)
            int numBins = N_FFT / 2 + 1;
            float[] real = new float[N_FFT];
            float[] imag = new float[N_FFT];

            for (int bin = 0; bin < numBins; bin++)
            {
                int idx = bin * numFrames + frame;
                float mag = magnitude[idx];
                float ph = phase[idx];
                real[bin] = mag * MathF.Cos(ph);
                imag[bin] = mag * MathF.Sin(ph);

                // 共役対称性
                if (bin > 0 && bin < numBins - 1)
                {
                    real[N_FFT - bin] = real[bin];
                    imag[N_FFT - bin] = -imag[bin];
                }
            }

            // 16点IFFT
            float[] timeSignal = IFFT16(real, imag);

            // オーバーラップ加算
            int frameStart = frame * HOP_LENGTH;
            for (int i = 0; i < N_FFT; i++)
            {
                if (frameStart + i < outputLen)
                {
                    output[frameStart + i] += timeSignal[i] * _window[i] / N_FFT;
                    windowSum[frameStart + i] += _window[i] * _window[i];
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
        // 16点IFFTの直接実装
        float[] result = new float[N_FFT];
        for (int n = 0; n < N_FFT; n++)
        {
            float sum = 0;
            for (int k = 0; k < N_FFT; k++)
            {
                float angle = 2 * MathF.PI * k * n / N_FFT;
                sum += real[k] * MathF.Cos(angle) - imag[k] * MathF.Sin(angle);
            }
            result[n] = sum / N_FFT;
        }
        return result;
    }
}
```

### 5.2 KVキャッシュ管理

```csharp
public class KVCacheManager
{
    private const int NUM_LAYERS = 24;
    private const int NUM_KV_HEADS = 2;
    private const int HEAD_DIM = 64;

    private Tensor<float>[] _keyCache;
    private Tensor<float>[] _valueCache;

    public KVCacheManager()
    {
        _keyCache = new Tensor<float>[NUM_LAYERS];
        _valueCache = new Tensor<float>[NUM_LAYERS];
    }

    public void InitFromInitialPass(IReadOnlyList<Tensor<float>> outputs)
    {
        // llm_backbone_initial の出力からKVキャッシュを初期化
        // 出力: hidden_states, past_key_0, past_value_0, ..., past_key_23, past_value_23
        for (int i = 0; i < NUM_LAYERS; i++)
        {
            _keyCache[i] = outputs[1 + i * 2];      // past_key_i
            _valueCache[i] = outputs[1 + i * 2 + 1]; // past_value_i
        }
    }

    public void UpdateFromDecodePass(IReadOnlyList<Tensor<float>> outputs)
    {
        // llm_backbone_decode の出力からKVキャッシュを更新
        for (int i = 0; i < NUM_LAYERS; i++)
        {
            _keyCache[i] = outputs[1 + i * 2];
            _valueCache[i] = outputs[1 + i * 2 + 1];
        }
    }

    public Dictionary<string, Tensor<float>> GetInputsForDecode()
    {
        var inputs = new Dictionary<string, Tensor<float>>();
        for (int i = 0; i < NUM_LAYERS; i++)
        {
            inputs[$"past_key_{i}"] = _keyCache[i];
            inputs[$"past_value_{i}"] = _valueCache[i];
        }
        return inputs;
    }
}
```

### 5.3 Euler Solver（Flow用）

```csharp
public class EulerSolver
{
    private const int NUM_STEPS = 10;

    public Tensor<float> Solve(
        IWorker estimator,
        Tensor<float> initialNoise,
        Tensor<float> mask,
        Tensor<float> mu,
        Tensor<float> spks,
        Tensor<float> cond)
    {
        Tensor<float> x = initialNoise;
        float dt = 1.0f / NUM_STEPS;

        for (int step = 0; step < NUM_STEPS; step++)
        {
            float t = step * dt;

            // velocity = estimator(x, mask, mu, t, spks, cond)
            var tTensor = new Tensor<float>(new[] { 1 }, new[] { t });

            estimator.SetInput("x", x);
            estimator.SetInput("mask", mask);
            estimator.SetInput("mu", mu);
            estimator.SetInput("t", tTensor);
            estimator.SetInput("spks", spks);
            estimator.SetInput("cond", cond);

            estimator.Schedule();
            var velocity = estimator.PeekOutput("estimator_out");

            // x = x + velocity * dt
            x = TensorOps.Add(x, TensorOps.Mul(velocity, dt));
        }

        return x;
    }
}
```

---

## 6. LLM推論フロー詳細

### 6.1 入力構成（Zero-Shotモード）

```
LLM入力 = [SOS, prompt_text_emb, tts_text_emb, TASK_ID, prompt_speech_token_emb]

- SOS: 開始トークン (ID: 1)
- prompt_text_emb: プロンプトテキストの埋め込み
- tts_text_emb: 合成テキストの埋め込み
- TASK_ID: タスクトークン (ID: 6759)
- prompt_speech_token_emb: プロンプト音声トークンの埋め込み
```

### 6.2 自己回帰ループ

```csharp
public List<int> GenerateSpeechTokens(
    Tensor<float> textEmbeds,
    Tensor<float> promptSpeechEmbeds,
    int maxLength = 2048)
{
    var speechTokens = new List<int>();
    var kvCache = new KVCacheManager();

    // 1. 初回パス（KVキャッシュ生成）
    var initialInput = ConcatEmbeddings(textEmbeds, promptSpeechEmbeds);
    var initialOutputs = _llmInitial.Run(initialInput);
    kvCache.InitFromInitialPass(initialOutputs);

    var hidden = initialOutputs[0]; // hidden_states

    // 2. デコードループ
    for (int i = 0; i < maxLength; i++)
    {
        // logits生成
        var logits = _llmDecoder.Run(hidden);

        // トークンサンプリング
        int token = SampleToken(logits);

        // EOS (ID: 2) で終了
        if (token == 2) break;

        speechTokens.Add(token);

        // 次のステップ用埋め込み
        var tokenEmbed = _speechEmbedding.Run(token);

        // デコードパス
        var decodeInputs = kvCache.GetInputsForDecode();
        decodeInputs["input_embeds"] = tokenEmbed;

        var decodeOutputs = _llmDecode.Run(decodeInputs);
        kvCache.UpdateFromDecodePass(decodeOutputs);

        hidden = decodeOutputs[0];
    }

    return speechTokens;
}
```

---

## 7. 特殊トークンID

| トークン | ID | 用途 |
|---------|-----|------|
| SOS | 1 | 開始トークン |
| EOS | 2 | 終了トークン |
| TASK_TOKEN | 6759 | Zero-Shotタスク識別 |
| Speech Token範囲 | 0-6760 | 音声トークン |

---

## 8. HiFTパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| n_fft | 16 | FFTウィンドウサイズ |
| hop_length | 4 | ホップ長 |
| upsample_rates | [8, 5, 3] | アップサンプル倍率（計120倍） |
| sample_rate | 24000 | 出力サンプルレート |

**STFTフレーム数の計算**:
```
stft_frames = mel_frames × 120 + 1
```

---

## 9. 参考リソース

### ドキュメント
- `docs/onnx-export-implementation.md` - ONNXエクスポート詳細
- `docs/onnx-inference-guide.md` - Python推論ガイド

### Pythonリファレンス実装
- `scripts/onnx_inference_pure.py` - PyTorchフリー推論スクリプト

### ZipVoice参考実装
- `C:\Users\yuta\Desktop\Private\uZipVoice` - Unity実装
- `C:\Users\yuta\Desktop\Private\ZipVoice\zipvoice\bin\onnx_export_sentis.py` - Sentisエクスポート

### Unity Sentisドキュメント
- [Supported ONNX operators](https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/supported-operators.html)
- [Unity AI Inference Engine](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.4/manual/index.html)

---

## 10. チェックリスト

### モデル準備
- [ ] flow.decoder.estimator を opset 15 で再エクスポート
- [ ] hift_source_generator を opset 15 で再エクスポート
- [ ] text_embedding を opset 15 で再エクスポート
- [ ] 全モデルの Sentis 互換性検証

### Unity実装
- [ ] ISTFT 実装（n_fft=16）
- [ ] STFT 実装（n_fft=16）
- [ ] Qwen2 トークナイザー実装
- [ ] KV キャッシュ管理実装
- [ ] Euler Solver 実装
- [ ] LLM 自己回帰ループ実装
- [ ] Flow 推論パイプライン実装
- [ ] HiFT 推論パイプライン実装

### テスト
- [ ] 各モデルの単体動作確認
- [ ] エンドツーエンド音声生成テスト
- [ ] パフォーマンス測定

# 参考プロジェクト技術仕様書

本ドキュメントは、CosyVoice3 Unity ONNX対応の参考となる3プロジェクトの詳細技術仕様をまとめたものです。

---

## 1. uZipVoice - Unity Flow Matching TTS

### 1.1 プロジェクト概要

- **目的:** ZipVoice (Flow Matching TTS) のUnity 6実装
- **モデルサイズ:** 123Mパラメータ
- **ONNXモデル数:** 3（text_encoder, fm_decoder, vocos）
- **サンプルレート:** 24kHz
- **ライセンス:** MIT (コード), Apache 2.0 (モデル)

### 1.2 ディレクトリ構造

```
Assets/uZipVoice/
├── Runtime/
│   ├── Audio/
│   │   ├── FeatureExtractor.cs      # メルスペクトログラム抽出
│   │   └── ISTFTProcessor.cs        # STFT → 波形変換
│   ├── Core/
│   │   ├── ZipVoiceManager.cs       # メインAPI、オーケストレーション
│   │   └── ZipVoiceConfig.cs        # 設定 (ScriptableObject)
│   ├── Inference/
│   │   ├── TextEncoder.cs           # ONNX: トークン → テキスト条件
│   │   ├── FMDecoder.cs             # ONNX: Flow Matchingデコーダー
│   │   ├── Vocos.cs                 # ONNX: メル → STFT
│   │   └── EulerSolver.cs           # ODE積分、タイムステップ計算
│   └── Tokenizer/
│       ├── EspeakTokenizer.cs       # G2P (espeak-ng経由)
│       ├── EspeakNative.cs          # P/Invoke定義
│       └── TokenMap.cs              # 音素 ↔ トークンIDマッピング
├── Samples/
├── Tests/                           # 97ユニットテスト
├── Models/                          # ONNXファイル
├── Plugins/
│   ├── NWaves.dll                   # 信号処理ライブラリ
│   └── Windows/x64/libespeak-ng.dll
└── Resources/
    └── tokens.txt                   # 音素 → トークンIDテーブル
```

### 1.3 推論パイプライン詳細

```
[ステージ1] トークナイズ
    Text → espeak-ng G2P → IPA音素 → Token IDs [1, T]

[ステージ2] 音声特徴抽出
    AudioClip → MelSpectrogram [T, 100]
    - FFTサイズ: 1024
    - ホップ長: 256
    - メルバンド数: 100
    - センターパディング: 反射パディング

[ステージ3] テキストエンコーディング
    Token IDs → text_encoder.onnx → Text Condition [1, T, 512]

[ステージ4] 音声条件構築
    Prompt Mel → Speech Condition [1, T, 512]
    - feat_scale: 0.1で正規化

[ステージ5] Flow Matchingデコーダー
    Euler Solver (8-16ステップ):
    for step in 0..num_steps:
        v = fm_decoder(t, x, text_cond, speech_cond, guidance)
        x = x + dt * v
    → Mel Features [1, T, 100]

[ステージ6] プロンプト領域トリミング
    生成メル[prompt_len:] のみ抽出

[ステージ7] メル転置
    [1, T, 100] → [1, 100, T] (Vocos入力形式)

[ステージ8] ボコーダー
    Mel → vocos.onnx → STFT係数
    - magnitude [1, 513, T]
    - phase_cos [1, 513, T]
    - phase_sin [1, 513, T]

[ステージ9] ISTFT
    STFT → ISTFTProcessor (NWaves) → 波形
    - オーバーラップ加算
    - COLA正規化
    - センターパディング除去

[ステージ10] 正規化・AudioClip生成
    波形正規化 → AudioClip (24kHz, mono)
```

### 1.4 メモリ管理パターン

**バッファ再利用:**
```csharp
private float[] _xBuffer;

public async UniTask<Tensor<float>> GenerateAsync(...)
{
    if (_xBuffer == null || _xBuffer.Length != totalSize)
        _xBuffer = new float[totalSize];

    // ガウスノイズ初期化 (Box-Muller変換)
    for (int i = 0; i < totalSize; i++)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        _xBuffer[i] = (float)Math.Sqrt(-2.0 * Math.Log(u1))
                      * (float)Math.Sin(2.0 * Math.PI * u2);
    }

    // Eulerループでバッファ再利用
    for (int step = 0; step < numSteps; step++)
    {
        // ...推論...
        for (int i = 0; i < totalSize; i++)
            _xBuffer[i] = xData[i] + dt * vData[i];

        x = new Tensor<float>(shape, _xBuffer);
    }
}
```

**Yield頻度削減:**
```csharp
// 4ステップごとにYield（16回 → 4回）
if (step % 4 == 0)
    await UniTask.Yield();
```

### 1.5 パフォーマンス特性

| メトリック | 値 | 備考 |
|-----------|-----|------|
| モデルサイズ (ONNX) | ~125MB | text_encoder (15MB) + fm_decoder (65MB) + vocos (45MB) |
| 初期化時間 | ~2-3秒 | ONNXロード + espeak-ngデータ |
| 合成時間 (1秒出力) | 25-30秒 | 16 Eulerステップ |
| 1ステップ時間 | ~1.5-2秒 | FMDecoder推論 + Euler計算 |
| ピークメモリ | ~800MB | ONNX + 中間テンソル |

---

## 2. ZipVoice - オリジナルFlow Matching TTS

### 2.1 プロジェクト概要

- **目的:** 高速・高品質ゼロショットTTS
- **モデルサイズ:** 123Mパラメータ
- **サポート言語:** 中国語、英語
- **サンプルレート:** 24kHz

### 2.2 モデルバリアント

| バリアント | 特徴 |
|-----------|------|
| ZipVoice | ベースモデル |
| ZipVoice-Distill | 蒸留版（高速推論） |
| ZipVoice-Dialog | マルチスピーカー対話 |
| ZipVoice-Dialog-Stereo | ステレオ対話生成 |

### 2.3 アーキテクチャ詳細

**Text Encoder (TTSZipformer):**
```
入力: トークン埋め込み
    ↓
入力投影: in_dim → encoder_dim (192)
    ↓
エンコーダースタック × 4:
    - Zipformer2EncoderLayer
    - マルチヘッドアテンション (4ヘッド)
    - CNNモジュール (カーネル: 9)
    - フィードフォワード (512)
    ↓
出力投影: encoder_dim → feat_dim (100)
```

**FM Decoder (TTSZipformer):**
```
入力: feat_dim*3 (メル + テキスト条件 + 音声条件)
    ↓
ブロック構成: [2, 2, 4, 4, 4] レイヤー
ダウンサンプリング: [1, 2, 4, 2, 1]
隠れ次元: 512
アテンションヘッド: 4
フィードフォワード: 1536
    ↓
出力: feat_dim (100)
```

### 2.4 ONNXエクスポート仕様

**標準エクスポート (`onnx_export.py`):**
```python
# text_encoder.onnx
torch.onnx.export(
    OnnxTextModel(model.text_encoder, ...),
    (tokens, prompt_tokens, prompt_features_len, speed),
    "text_encoder.onnx",
    input_names=["tokens", "prompt_tokens", "prompt_features_len", "speed"],
    output_names=["text_condition"],
    dynamic_axes={
        "tokens": {0: "batch", 1: "seq_len"},
        "prompt_tokens": {0: "batch", 1: "prompt_seq_len"},
        "text_condition": {0: "batch", 1: "time"}
    },
    opset_version=13
)

# fm_decoder.onnx
torch.onnx.export(
    OnnxFlowMatchingModel(model.decoder, ...),
    (t, x, text_condition, speech_condition, guidance_scale),
    "fm_decoder.onnx",
    input_names=["t", "x", "text_condition", "speech_condition", "guidance_scale"],
    output_names=["v"],
    dynamic_axes={
        "x": {0: "batch", 1: "time"},
        "text_condition": {0: "batch", 1: "time"},
        "speech_condition": {0: "batch", 1: "time"}
    },
    opset_version=13
)
```

**Sentis専用エクスポート (`onnx_export_sentis.py`):**
- Opset: 15 (Sentis推奨)
- 定数畳み込み: 有効
- 位置エンコーディング: 事前計算 (max_len=4000)
- メタデータ埋め込み

### 2.5 Classifier-Free Guidance実装

```python
class OnnxFlowMatchingModel(nn.Module):
    def forward(self, t, x, text_condition, speech_condition, guidance_scale):
        if guidance_scale == 0:
            return self.model(...)

        # 条件付き/無条件の2ブランチ
        x_dup = x.repeat(2, 1, 1)
        text_cond_dup = torch.cat([
            torch.zeros_like(text_condition),
            text_condition
        ], dim=0)

        # タイムステップに応じた音声条件処理
        if t > 0.5:
            speech_cond_dup = torch.cat([
                torch.zeros_like(speech_condition),
                speech_condition
            ], dim=0)
            scale = guidance_scale
        else:
            speech_cond_dup = torch.cat([
                speech_condition,
                speech_condition
            ], dim=0)
            scale = guidance_scale * 2

        output = self.model(t, x_dup, text_cond_dup, speech_cond_dup)
        data_uncond, data_cond = output.chunk(2, dim=0)

        # CFG: v = (1 + scale) * v_cond - scale * v_uncond
        v = (1 + scale) * data_cond - scale * data_uncond
        return v
```

### 2.6 INT8量子化

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="text_encoder.onnx",
    model_output="text_encoder_int8.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul"]
)

# 効果:
# - サイズ: ~25%削減
# - 速度: 10-20%向上
# - 精度: 軽微な低下
```

---

## 3. uPiper - Unity Piper TTS

### 3.1 プロジェクト概要

- **目的:** Piper TTS (VITS) のUnity実装
- **アーキテクチャ:** VITS (Variational Inference TTS)
- **サポート言語:** 日本語、英語
- **サンプルレート:** 22050Hz

### 3.2 音素変換システム

**日本語 (OpenJTalk):**
```csharp
// P/Invoke定義
[DllImport("openjtalk_wrapper")]
private static extern int openjtalk_initialize(
    string dic_path,
    string user_dic_path);

[DllImport("openjtalk_wrapper")]
private static extern IntPtr openjtalk_phonemize(
    IntPtr handle,
    string text);

[DllImport("openjtalk_wrapper")]
private static extern IntPtr openjtalk_phonemize_with_prosody(
    IntPtr handle,
    string text);

// 使用例
var phonemes = openjtalk_phonemize(handle, "こんにちは");
// 結果: ["k", "o", "N", "n", "i", "ch", "i", "w", "a"]
```

**英語 (Flite LTS):**
- 純粋C#実装
- Letter-to-Sound規則エンジン
- 辞書ルックアップ + ルールフォールバック

### 3.3 PUAエンコーディング詳細

```csharp
private static readonly Dictionary<string, string> multiCharPhonemeMap = new()
{
    // 口蓋化子音
    ["ky"] = "\ue006",   // きゃ行
    ["gy"] = "\ue008",   // ぎゃ行
    ["ty"] = "\ue00a",   // ちゃ行 (ty)
    ["dy"] = "\ue00b",   // ぢゃ行
    ["py"] = "\ue00c",   // ぴゃ行
    ["by"] = "\ue00d",   // びゃ行
    ["hy"] = "\ue012",   // ひゃ行
    ["ny"] = "\ue013",   // にゃ行
    ["my"] = "\ue014",   // みゃ行
    ["ry"] = "\ue015",   // りゃ行

    // 破擦音・摩擦音
    ["ch"] = "\ue00e",   // ち、ちゃ行
    ["ts"] = "\ue00f",   // つ
    ["sh"] = "\ue010",   // し、しゃ行
    ["zy"] = "\ue011",   // じゃ行

    // 長母音
    ["a:"] = "\ue000",   // あー
    ["i:"] = "\ue001",   // いー
    ["u:"] = "\ue002",   // うー
    ["e:"] = "\ue003",   // えー
    ["o:"] = "\ue004",   // おー
};
```

### 3.4 推論パイプライン

```csharp
public async Task<AudioClip> GenerateAudioWithInferenceAsync(
    string text,
    float lengthScale = 1.0f,
    float noiseScale = 0.667f,
    float noiseW = 0.8f,
    CancellationToken cancellationToken = default)
{
    // 1. 音素変換
    var phonemeResult = await GetPhonemesAsync(text, cancellationToken);

    // 2. 音素 → トークンID
    var phonemeIds = _phonemeEncoder.Encode(phonemeResult.Phonemes);

    // 3. VITS推論（メインスレッド必須）
    var audioData = await UnityMainThreadDispatcher.RunOnMainThreadAsync(() =>
    {
        lock (_lockObject)
        {
            // 入力テンソル作成
            var inputTensor = new Tensor<int>(
                new TensorShape(1, phonemeIds.Length),
                phonemeIds);
            var lengthsTensor = new Tensor<int>(
                new TensorShape(1),
                new[] { phonemeIds.Length });
            var scalesTensor = new Tensor<float>(
                new TensorShape(3),
                new[] { noiseScale, lengthScale, noiseW });

            // 入力設定
            _worker.SetInput("input", inputTensor);
            _worker.SetInput("input_lengths", lengthsTensor);
            _worker.SetInput("scales", scalesTensor);

            // 推論実行
            _worker.Schedule();

            // 出力取得
            var outputTensor = _worker.PeekOutput<float>();
            var audio = new float[outputTensor.shape.length];
            for (int i = 0; i < audio.Length; i++)
                audio[i] = outputTensor[i];

            return audio;
        }
    });

    // 4. 正規化・AudioClip生成
    var normalizedAudio = _audioClipBuilder.NormalizeAudio(audioData, 0.95f);
    return _audioClipBuilder.BuildAudioClip(normalizedAudio, 22050, "TTS_output");
}
```

### 3.5 韻律サポート（オプション）

```csharp
// 韻律サポートモデルの場合
if (_generator.SupportsProsody)
{
    // A1: アクセント核位置
    // A2: アクセント句境界
    // A3: 呼気段落境界
    var prosodyData = new float[sequenceLength * 3];
    for (int i = 0; i < sequenceLength; i++)
    {
        prosodyData[i * 3 + 0] = prosodyA1[i];
        prosodyData[i * 3 + 1] = prosodyA2[i];
        prosodyData[i * 3 + 2] = prosodyA3[i];
    }

    var prosodyTensor = new Tensor<float>(
        new TensorShape(1, sequenceLength, 3),
        prosodyData);
    _worker.SetInput("prosody_features", prosodyTensor);
}
```

### 3.6 バックエンド選択ロジック

```csharp
private BackendType DetermineBackendType(PiperConfig config)
{
    // Metalの問題回避
    if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Metal)
        return BackendType.CPU;

    // GPUComputeはVITSで問題あり
    if (config.Backend == InferenceBackend.GPUCompute)
        return BackendType.GPUPixel;

    // プラットフォーム別デフォルト
    #if UNITY_WEBGL
        return BackendType.GPUPixel;
    #elif UNITY_IOS || UNITY_ANDROID
        return BackendType.GPUPixel;
    #else
        return BackendType.GPUPixel;
    #endif
}
```

### 3.7 パフォーマンス特性

| ステージ | 時間 |
|---------|------|
| 音素変換 | 5-10ms |
| 音素エンコーディング | 1-2ms |
| ONNX推論 (CPU) | 300-800ms |
| ONNX推論 (GPU) | 50-200ms |
| 音声正規化 | 1-5ms |
| **合計** | **357-1017ms** |

| コンポーネント | サイズ |
|--------------|--------|
| OpenJTalkライブラリ | 15-20 MB |
| モデル (ONNX) | 40-100 MB |
| 辞書 (mecab) | 20-30 MB |
| ランタイムヒープ | 50-200 MB |
| **合計** | **125-350 MB** |

---

## 4. 共通技術パターン

### 4.1 Unity Sentis制約対応

| 制約 | uZipVoice対応 | uPiper対応 |
|-----|---------------|------------|
| FFT未サポート | NWavesでISTFT実装 | N/A (VITSは直接波形出力) |
| スカラーランク0 | TensorShape()使用 | TensorShape()使用 |
| メインスレッド必須 | UniTask.Yield() | UnityMainThreadDispatcher |
| 動的形状 | dynamic_axes指定 | dynamic_axes指定 |

### 4.2 AudioClip生成パターン

```csharp
// 共通パターン
AudioClip clip = AudioClip.Create(
    name: "Synthesized",
    lengthSamples: audioData.Length,
    channels: 1,               // モノラル
    frequency: sampleRate,     // 22050 or 24000
    stream: false);

clip.SetData(audioData, 0);

// 正規化
float maxAmp = audioData.Max(x => Math.Abs(x));
float scale = 0.95f / maxAmp;
for (int i = 0; i < audioData.Length; i++)
    audioData[i] *= scale;
```

### 4.3 テンソル操作パターン

```csharp
// 1D: トークンID
var ids = new Tensor<int>(new TensorShape(seqLen), idArray);

// 2D: バッチ
var batch = new Tensor<float>(new TensorShape(batchSize, features), data);

// 3D: シーケンス特徴
var seq = new Tensor<float>(new TensorShape(1, seqLen, featDim), data);

// スカラー（ランク0）
var scalar = new Tensor<float>(new TensorShape(), new[] { value });
```

### 4.4 メモリ管理パターン

```csharp
// using文でテンソル自動破棄
using var inputTensor = new Tensor<int>(shape, data);
using var outputTensor = _worker.PeekOutput() as Tensor<float>;

// 手動破棄
tensor.Dispose();

// IDisposable実装
public class Inference : IDisposable
{
    private Worker _worker;

    public void Dispose()
    {
        _worker?.Dispose();
        _worker = null;
    }
}
```

---

## 5. CosyVoice3適用への示唆

### 5.1 流用可能なコンポーネント

| コンポーネント | ソース | 適用箇所 |
|--------------|--------|----------|
| EulerSolver | uZipVoice | Flow Matchingステップ |
| ISTFTProcessor | uZipVoice | HiFi-GANがISTFT使用の場合 |
| FeatureExtractor | uZipVoice | プロンプト音声処理 |
| OpenJTalk P/Invoke | uPiper | 日本語音素変換 |
| AudioClipBuilder | uPiper | 音声出力 |
| スレッド安全パターン | uPiper | メインスレッド同期 |

### 5.2 CosyVoice3固有の課題

1. **LLMサイズ**
   - ZipVoice Text Encoder: 15MB
   - CosyVoice3 LLM: ~400MB (推定)
   - 対策: 量子化、蒸留、またはサーバーオフロード

2. **自己回帰生成**
   - ZipVoice: 非自己回帰
   - CosyVoice3: LLMが自己回帰
   - 対策: KVキャッシュ管理、バッチ処理最適化

3. **DiT構造**
   - ZipVoice: Zipformerベース
   - CosyVoice3: Diffusion Transformer
   - 対策: DiT専用のONNXラッパー作成

### 5.3 推奨実装順序

1. **HiFi-GAN (HIFT)** - 最も単純、検証容易
2. **Flow Matching + DiT** - uZipVoiceパターン適用
3. **音声トークナイザー** - プロンプトエンコーディング
4. **LLM** - 最大の課題、慎重に検討

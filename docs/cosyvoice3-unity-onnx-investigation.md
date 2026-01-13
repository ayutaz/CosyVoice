# CosyVoice3 Unity ONNX対応調査レポート

## 概要

本ドキュメントはCosyVoice3をONNXエクスポートしてUnity AI Interface (Sentis)で動作させるための調査結果をまとめたものです。

### 調査対象プロジェクト

| プロジェクト | パス | 概要 |
|-------------|------|------|
| uZipVoice | `C:\Users\yuta\Desktop\Private\uZipVoice` | ZipVoice (Flow Matching TTS)のUnity実装 |
| ZipVoice | `C:\Users\yuta\Desktop\Private\ZipVoice` | オリジナルFlow Matching TTS |
| uPiper | `C:\Users\yuta\Desktop\Private\uPiper` | Piper TTS (VITS)のUnity実装 |

---

## 1. アーキテクチャ比較

### 1.1 CosyVoice3 アーキテクチャ

```
テキスト入力
    ↓
[Frontend] テキスト正規化・言語タグ付与
    ↓
[Tokenizer] テキスト → トークンID
    ↓
[LLM (Qwen)] テキストトークン → 音声トークン  ← 最大の課題
    ↓
[Flow Matching + DiT] 音声トークン → メルスペクトログラム
    ↓
[HiFi-GAN (HIFT)] メルスペクトログラム → 波形 (24kHz)
```

**モデルサイズ:**
- Fun-CosyVoice3-0.5B: 約500Mパラメータ
- LLM部分: 約400M（大部分を占める）
- Flow Matching: 約80M
- HiFi-GAN: 約20M

### 1.2 ZipVoice アーキテクチャ（参考実装）

```
テキスト入力
    ↓
[Tokenizer (espeak-ng)] テキスト → IPA音素 → トークンID
    ↓
[Text Encoder] トークン → テキスト条件ベクトル [512D]
    ↓
[FM Decoder + Euler Solver] Flow Matching (8-16ステップ)
    ↓
[Vocos] メル特徴 → STFT係数
    ↓
[ISTFT] STFT → 波形 (24kHz)
```

**モデルサイズ:**
- Text Encoder: ~15MB
- FM Decoder: ~65MB
- Vocos: ~45MB
- **合計: ~125MB** (ONNXでUnity動作可能)

### 1.3 主要な違い

| 項目 | CosyVoice3 | ZipVoice |
|------|-----------|----------|
| テキスト処理 | LLM (Qwen-based, 400M) | Text Encoder (15M) |
| 音声生成 | Flow Matching + DiT | Flow Matching |
| ボコーダー | HiFi-GAN (HIFT) | Vocos + ISTFT |
| 合計サイズ | ~500MB | ~125MB |
| ゼロショット | 強力（プロンプト音声） | 対応 |
| 多言語 | 9言語+18方言 | 中英のみ |

---

## 2. Unity ONNX実装パターン（uZipVoice分析）

### 2.1 ONNXモデル構成

uZipVoiceは3つのONNXモデルを使用:

```
Assets/uZipVoice/Models/
├── text_encoder.onnx      # テキスト → 条件ベクトル
├── fm_decoder.onnx        # Flow Matching デコーダー
└── vocos_opset15.onnx     # メル → STFT係数
```

### 2.2 Text Encoder ONNX仕様

**入力:**
| 名前 | 形状 | 型 | 備考 |
|------|------|-----|------|
| tokens | [N, T] | INT64 | 音素トークンID |
| prompt_tokens | [N, T_p] | INT64 | 参照テキストトークン |
| prompt_features_len | スカラー | INT64 | **ランク0テンソル** |
| speed | スカラー | FLOAT | **ランク0テンソル** |

**出力:**
| 名前 | 形状 | 型 |
|------|------|-----|
| text_condition | [N, T, 512] | FLOAT |

### 2.3 FM Decoder ONNX仕様

**入力:**
| 名前 | 形状 | 型 | 備考 |
|------|------|-----|------|
| t | スカラー | FLOAT | タイムステップ (0-1) |
| x | [N, T, 100] | FLOAT | 現在状態（メル特徴） |
| text_condition | [N, T, 512] | FLOAT | テキスト条件 |
| speech_condition | [N, T, 100] | FLOAT | 音声条件（プロンプト） |
| guidance_scale | スカラー | FLOAT | CFGスケール |

**出力:**
| 名前 | 形状 | 型 |
|------|------|-----|
| velocity | [N, T, 100] | FLOAT |

### 2.4 Vocos ONNX仕様

**入力:**
| 名前 | 形状 | 型 | 備考 |
|------|------|-----|------|
| mel_spectrogram | [N, 100, T] | FLOAT | **[N, MEL, TIME]** 形式 |

**出力:**
| 名前 | 形状 | 型 | 備考 |
|------|------|-----|------|
| magnitude | [N, 513, T] | FLOAT | STFT振幅 |
| phase_cos | [N, 513, T] | FLOAT | cos(位相) |
| phase_sin | [N, 513, T] | FLOAT | sin(位相) |

### 2.5 Unity Sentis制約事項

| 制約 | 影響 | 解決策 |
|-----|------|--------|
| 最大テンソル次元 | 8Dまで | 全テンソル ≤ 8D ✓ |
| 未サポート演算子 | FFT, IFFT, RFFT, IRFFT | C#でISTFT実装 (NWaves) |
| Opsetバージョン | 7-15のみ | opset 15使用 |
| スカラーテンソル | ランク0必須（shape [1]不可） | TensorShape()で作成 |
| 動的形状 | 限定サポート | dynamic_axes指定 |

### 2.6 スカラーテンソル作成の重要ポイント

```csharp
// ❌ 間違い - shape [1] はSentisで拒否される
using var speedTensor = new Tensor<float>(
    new TensorShape(1),
    new float[] { speed }
);

// ✅ 正解 - ランク0テンソル
using var speedTensor = new Tensor<float>(
    new TensorShape(),  // ← 引数なし = ランク0
    new float[] { speed }
);
```

---

## 3. ZipVoice ONNXエクスポート手法

### 3.1 エクスポートスクリプト

**場所:** `zipvoice/bin/onnx_export.py`, `zipvoice/bin/onnx_export_sentis.py`

**標準エクスポート:**
```python
torch.onnx.export(
    model.text_encoder,
    (tokens, prompt_tokens, prompt_features_len, speed),
    "text_encoder.onnx",
    input_names=["tokens", "prompt_tokens", "prompt_features_len", "speed"],
    output_names=["text_condition"],
    dynamic_axes={
        "tokens": {0: "batch", 1: "seq_len"},
        "prompt_tokens": {0: "batch", 1: "prompt_seq_len"},
        "text_condition": {0: "batch", 1: "time"}
    },
    opset_version=15
)
```

**Sentis専用エクスポート:**
- Opset 15（Sentis推奨）
- 定数畳み込み有効化
- 位置エンコーディング事前計算（max_len=4000フレーム）
- 互換性検証付き

### 3.2 ラッパークラス

**OnnxTextModel:**
```python
class OnnxTextModel(nn.Module):
    def forward(self, tokens, prompt_tokens, prompt_features_len, speed):
        # 1. prompt_tokens + tokens を結合
        # 2. パディング処理
        # 3. 埋め込み層処理
        # 4. テキストエンコーダー処理
        # 5. 特徴長計算・繰り返し処理
        return text_condition
```

**OnnxFlowMatchingModel:**
```python
class OnnxFlowMatchingModel(nn.Module):
    def forward(self, t, x, text_condition, speech_condition, guidance_scale):
        # Classifier-Free Guidance処理
        if guidance_scale == 0:
            return model(...)
        else:
            # 条件付き/無条件の2ブランチ処理
            v = (1 + scale) * v_cond - scale * v_uncond
            return v
```

### 3.3 INT8量子化

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="text_encoder.onnx",
    model_output="text_encoder_int8.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul"]
)
```

- モデルサイズ: ~25%削減
- 推論速度: 10-20%向上

---

## 4. uPiper Unity実装パターン

### 4.1 音素変換システム

**日本語 (OpenJTalk):**
```csharp
[DllImport("openjtalk_wrapper")]
private static extern IntPtr openjtalk_phonemize(IntPtr handle, string text);

// 例: "こんにちは" → ["k", "o", "N", "n", "i", "ch", "i", "w", "a"]
```

**英語 (Flite LTS):**
- 純粋C#実装のLetter-to-Sound規則
- 辞書ルックアップ + ルールベースフォールバック

### 4.2 PUA (Private Use Area) エンコーディング

多文字音素をUnicode私用領域にマッピング:

```csharp
private static readonly Dictionary<string, string> multiCharPhonemeMap = new()
{
    ["ky"] = "\ue006",  // きゃ、きゅ、きょ
    ["gy"] = "\ue008",  // ぎゃ、ぎゅ、ぎょ
    ["ch"] = "\ue00e",  // ち、ちゃ、ちゅ、ちょ
    ["ts"] = "\ue00f",  // つ
    ["sh"] = "\ue010",  // し、しゃ、しゅ、しょ
    // ...
};
```

### 4.3 スレッド安全性

```csharp
// Unity.InferenceEngineはメインスレッド必須
await UnityMainThreadDispatcher.RunOnMainThreadAsync(() => {
    lock (_lockObject) {
        _worker.SetInput("input", inputTensor);
        _worker.Schedule();
        // ...
    }
});
```

### 4.4 オーディオ生成

```csharp
public AudioClip BuildAudioClip(float[] audioData, int sampleRate, string clipName)
{
    var audioClip = AudioClip.Create(
        name: clipName,
        lengthSamples: audioData.Length,
        channels: 1,  // モノラル
        frequency: sampleRate,
        stream: false);

    audioClip.SetData(audioData, 0);
    return audioClip;
}
```

---

## 5. CosyVoice3 ONNXエクスポート戦略

### 5.1 課題分析

**主要課題:**

1. **LLMサイズ問題**
   - Qwen系LLM: ~400Mパラメータ
   - ONNXファイルサイズ: 推定1.5GB以上
   - Unityメモリ制約に抵触

2. **自己回帰生成**
   - LLMは1トークンずつ生成
   - Sentisでのループ処理に制約

3. **KVキャッシュ**
   - 効率的な推論にKVキャッシュ必要
   - 動的サイズ管理の複雑さ

### 5.2 提案アプローチ

#### アプローチA: モデル分割

```
[Frontend + Tokenizer] → C#実装
[LLM] → ONNXエクスポート（大きいが必須）
[Flow Matching] → ONNXエクスポート
[HiFi-GAN] → ONNXエクスポート
```

**利点:** フル機能維持
**欠点:** LLMサイズが巨大

#### アプローチB: 蒸留モデル作成

1. CosyVoice3 LLM → 小型蒸留モデル (~50-100M)
2. Flow Matching → そのままエクスポート
3. HiFi-GAN → そのままエクスポート

**利点:** サイズ削減
**欠点:** 品質低下の可能性、追加学習必要

#### アプローチC: サーバーハイブリッド

```
[Unity側]
- 音声プロンプトエンコーディング
- メルスペクトログラム → 波形 (HiFi-GAN)

[サーバー側]
- LLM推論
- Flow Matching
```

**利点:** Unity側の負荷軽減
**欠点:** ネットワーク依存、オフライン不可

### 5.3 推奨実装ステップ

**Phase 1: 基盤調査**
1. CosyVoice3の各コンポーネント詳細分析
2. 既存エクスポートスクリプト (`cosyvoice/bin/export_onnx.py`) の確認
3. 各モジュールの入出力仕様の特定

**Phase 2: ONNXエクスポート**
1. HiFi-GAN (HIFT) のONNXエクスポート（最も単純）
2. Flow Matching + DiTのONNXエクスポート
3. LLMのONNXエクスポート（課題あり）

**Phase 3: Unity実装**
1. uZipVoiceをテンプレートとして使用
2. EulerSolver、ISTFTProcessor等を流用
3. CosyVoice3固有の処理を追加

**Phase 4: 最適化**
1. INT8量子化の適用
2. モデルプルーニング検討
3. TensorRT対応（Linux GPU環境向け）

---

## 6. 既存CosyVoice ONNXエクスポート調査

### 6.1 エクスポートスクリプト

**場所:** `cosyvoice/bin/export_onnx.py`

```python
# 主要エクスポート対象
- speech_tokenizer_session: 音声 → トークン
- llm: テキスト → 音声トークン
- flow: トークン → メルスペクトログラム
- hift: メルスペクトログラム → 波形
```

### 6.2 モデルコンポーネント

**CosyVoice3Model クラス (`cosyvoice/cli/model.py`):**

```python
class CosyVoice3Model:
    def __init__(self):
        self.llm = ...           # Qwen系LLM
        self.flow = ...          # Flow Matching + DiT
        self.hift = ...          # HiFi-GAN ボコーダー
        self.speech_tokenizer = ...  # 音声トークナイザー
```

### 6.3 推論フロー

```python
def inference_cross_lingual(self, tts_text, prompt_speech_16k):
    # 1. プロンプト音声をエンコード
    prompt_speech_token = self.speech_tokenizer_session(prompt_speech_16k)
    prompt_speech_feat = self.frontend.frontend_cross_lingual(...)

    # 2. LLM推論（自己回帰）
    for token in self.llm.inference(
        text=tts_text,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_feat=prompt_speech_feat
    ):
        # トークンをストリーミング生成

    # 3. Flow Matching
    mel = self.flow.inference(speech_token=speech_token)

    # 4. HiFi-GAN
    audio = self.hift.inference(mel=mel)

    return audio
```

---

## 7. 次のステップ

### 7.1 即座に実行可能な作業

1. **HiFi-GAN (HIFT) のONNXエクスポート検証**
   - 入出力形状の確認
   - Sentis互換性テスト

2. **Flow Matchingのエクスポート検証**
   - DiT部分の処理確認
   - EulerSolver互換性確認

3. **LLMエクスポートの試行**
   - サイズ・メモリ要件の確認
   - 自己回帰推論のONNX化可否

### 7.2 中期的な作業

1. **Unity側プロトタイプ作成**
   - uZipVoiceをフォークしてベース作成
   - CosyVoice3用に調整

2. **パフォーマンス計測**
   - 各モデルの推論時間
   - メモリ使用量

3. **最適化検討**
   - 量子化効果の測定
   - モデルサイズ削減手法

---

## 8. 参考実装コード例

### 8.1 Euler Solver (C#)

```csharp
public class EulerSolver
{
    private float[] _timesteps;
    private int _numSteps;
    private float _tShift;

    public EulerSolver(int numSteps, float tShift = 0.5f)
    {
        _numSteps = numSteps;
        _tShift = tShift;
        _timesteps = ComputeTimesteps();
    }

    private float[] ComputeTimesteps()
    {
        var timesteps = new float[_numSteps + 1];
        for (int i = 0; i <= _numSteps; i++)
        {
            float t = (float)i / _numSteps;
            float tShifted = _tShift * t / (1f + (_tShift - 1f) * t);
            timesteps[i] = tShifted;
        }
        return timesteps;
    }

    public float GetTimestep(int index) => _timesteps[index];
    public float GetDt(int stepIndex) => _timesteps[stepIndex + 1] - _timesteps[stepIndex];
}
```

### 8.2 ISTFT Processor (C#)

```csharp
public class ISTFTProcessor
{
    private int _nFft = 1024;
    private int _hopLength = 256;
    private float[] _window;
    private Fft _fft;  // NWaves FFT

    public float[] Process(float[] magnitude, float[] phaseCos,
                           float[] phaseSin, int numBins, int numFrames)
    {
        int fullLength = (numFrames - 1) * _hopLength + _nFft;
        float[] output = new float[fullLength];
        float[] windowSum = new float[fullLength];

        for (int frame = 0; frame < numFrames; frame++)
        {
            // STFT係数から複素スペクトル再構成
            float[] realSpectrum = new float[_nFft];
            float[] imagSpectrum = new float[_nFft];

            for (int bin = 0; bin < numBins; bin++)
            {
                int idx = bin * numFrames + frame;
                float mag = magnitude[idx];
                float cos = phaseCos[idx];
                float sin = phaseSin[idx];

                realSpectrum[bin] = mag * cos;
                imagSpectrum[bin] = mag * sin;

                // 共役対称性
                if (bin > 0 && bin < numBins - 1)
                {
                    int mirrorBin = _nFft - bin;
                    realSpectrum[mirrorBin] = mag * cos;
                    imagSpectrum[mirrorBin] = -mag * sin;
                }
            }

            // IFFT
            _fft.Inverse(realSpectrum, imagSpectrum);

            // オーバーラップ加算
            int frameStart = frame * _hopLength;
            float normFactor = 1.0f / _nFft;
            for (int i = 0; i < _nFft; i++)
            {
                int outIdx = frameStart + i;
                if (outIdx < fullLength)
                {
                    output[outIdx] += realSpectrum[i] * normFactor * _window[i];
                    windowSum[outIdx] += _window[i] * _window[i];
                }
            }
        }

        // COLA正規化
        for (int i = 0; i < fullLength; i++)
        {
            if (windowSum[i] > 1e-8f)
                output[i] /= windowSum[i];
        }

        // センターパディング除去
        int pad = _nFft / 2;
        int trimmedLength = fullLength - 2 * pad;
        float[] result = new float[trimmedLength];
        Array.Copy(output, pad, result, 0, trimmedLength);

        return result;
    }
}
```

### 8.3 Sentis推論パターン

```csharp
using Unity.InferenceEngine;

public class CosyVoiceInference : IDisposable
{
    private Model _flowModel;
    private Worker _flowWorker;

    public async Task InitializeAsync(ModelAsset flowModelAsset)
    {
        _flowModel = ModelLoader.Load(flowModelAsset);
        _flowWorker = new Worker(_flowModel, BackendType.GPUPixel);
    }

    public async Task<float[]> GenerateFlowAsync(
        Tensor<float> speechToken,
        int numSteps = 8)
    {
        var solver = new EulerSolver(numSteps, tShift: 0.5f);

        // ノイズ初期化
        var shape = new TensorShape(1, speechToken.shape[1], 100);
        var x = GenerateGaussianNoise(shape);

        // Eulerステップ
        for (int step = 0; step < numSteps; step++)
        {
            float t = solver.GetTimestep(step);
            float dt = solver.GetDt(step);

            // スカラーテンソル（ランク0）
            using var tTensor = new Tensor<float>(new TensorShape(), new[] { t });

            _flowWorker.SetInput("t", tTensor);
            _flowWorker.SetInput("x", x);
            _flowWorker.SetInput("speech_token", speechToken);
            _flowWorker.Schedule();

            var velocity = _flowWorker.PeekOutput("v") as Tensor<float>;

            // Eulerステップ: x = x + dt * v
            x = EulerStep(x, velocity, dt);

            if (step % 4 == 0)
                await Task.Yield();
        }

        return x.DownloadToArray();
    }

    public void Dispose()
    {
        _flowWorker?.Dispose();
    }
}
```

---

## 9. 結論

### 9.1 実現可能性評価

| コンポーネント | Unity移植難易度 | 備考 |
|--------------|----------------|------|
| HiFi-GAN (HIFT) | 低 | 単純なCNN、エクスポート容易 |
| Flow Matching | 中 | DiT構造あり、要検証 |
| LLM (Qwen) | 高 | サイズ大、自己回帰必要 |
| 音素変換 | 中 | OpenJTalk P/Invoke流用可能 |

### 9.2 推奨アプローチ

**短期目標:** HiFi-GAN + Flow MatchingのONNXエクスポートと検証

**中期目標:** LLMの小型化または代替アーキテクチャの検討

**長期目標:** フルパイプラインのUnity統合

### 9.3 参照プロジェクト活用

- **uZipVoice:** Unity Sentis統合のベストプラクティス
- **ZipVoice:** ONNXエクスポートスクリプトのテンプレート
- **uPiper:** 日本語音素変換・スレッド安全性パターン

---

## 付録

### A. ファイルパス一覧

**uZipVoice主要ファイル:**
- `C:\Users\yuta\Desktop\Private\uZipVoice\Assets\uZipVoice\Runtime\Core\ZipVoiceManager.cs`
- `C:\Users\yuta\Desktop\Private\uZipVoice\Assets\uZipVoice\Runtime\Inference\TextEncoder.cs`
- `C:\Users\yuta\Desktop\Private\uZipVoice\Assets\uZipVoice\Runtime\Inference\FMDecoder.cs`
- `C:\Users\yuta\Desktop\Private\uZipVoice\Assets\uZipVoice\Runtime\Inference\Vocos.cs`
- `C:\Users\yuta\Desktop\Private\uZipVoice\Assets\uZipVoice\Runtime\Inference\EulerSolver.cs`
- `C:\Users\yuta\Desktop\Private\uZipVoice\Assets\uZipVoice\Runtime\Audio\ISTFTProcessor.cs`

**ZipVoice主要ファイル:**
- `C:\Users\yuta\Desktop\Private\ZipVoice\zipvoice\bin\onnx_export.py`
- `C:\Users\yuta\Desktop\Private\ZipVoice\zipvoice\bin\onnx_export_sentis.py`
- `C:\Users\yuta\Desktop\Private\ZipVoice\zipvoice\models\zipvoice.py`

**uPiper主要ファイル:**
- `C:\Users\yuta\Desktop\Private\uPiper\Assets\uPiper\Runtime\Core\PiperTTS.cs`
- `C:\Users\yuta\Desktop\Private\uPiper\Assets\uPiper\Runtime\Core\AudioGeneration\InferenceAudioGenerator.cs`
- `C:\Users\yuta\Desktop\Private\uPiper\Assets\uPiper\Runtime\Core\Phonemizers\OpenJTalkPhonemizer.cs`

### B. 用語集

| 用語 | 説明 |
|------|------|
| Sentis | Unity AI Inference Engine（旧Barracuda後継） |
| Flow Matching | 確率的フロー生成モデル（拡散モデルより高速） |
| DiT | Diffusion Transformer |
| VITS | Variational Inference with adversarial learning for end-to-end TTS |
| HiFi-GAN | High-Fidelity GAN ボコーダー |
| ISTFT | Inverse Short-Time Fourier Transform |
| CFG | Classifier-Free Guidance |
| PUA | Private Use Area（Unicode私用領域） |

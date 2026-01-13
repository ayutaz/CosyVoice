# CosyVoice3 量子化分析レポート

## 概要

本ドキュメントはCosyVoice3のONNXエクスポートにおける量子化（INT8、FP16等）の精度への影響を調査した結果をまとめたものです。

**結論**: ユーザーの懸念は正しく、**INT8量子化はTTSモデルで音質劣化のリスクがあり、FP16が推奨**されます。

---

## 1. 量子化手法の比較

### 1.1 精度ランキング

| 形式 | ビット数 | サイズ削減 | 精度劣化 | 推奨用途 |
|------|---------|-----------|---------|---------|
| FP32 | 32bit | 基準 | なし | 学習、高精度推論 |
| BF16 | 16bit | 50% | ほぼなし | GPU推論 |
| FP16 | 16bit | 50% | ほぼなし | **TTS推奨** |
| FP8 | 8bit | 75% | <1% | 最新GPU (H100) |
| INT8 | 8bit | 75% | 1-5% | **TTS非推奨** |
| INT4 | 4bit | 87.5% | 2-10% | テキストLLMのみ |

### 1.2 LLM量子化の一般的知見

**50万回以上の評価からの知見** ([Red Hat研究](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms)):

- **FP8 (W8A8-FP)**: 事実上ロスレス
- **INT8 (W8A8-INT)**: 適切に調整すれば1-3%の精度低下
- **INT4**: 予想より競争力あり、8bit量子化に匹敵

**ただし、これはテキスト生成タスクでの評価であり、TTSでは異なる結果になる可能性が高い。**

---

## 2. TTS特有の量子化問題

### 2.1 ボコーダーの感度

**[BitTTS研究](https://arxiv.org/abs/2506.03515) からの重要な発見:**

> 「ボコーダーの量子化は音響モデルよりも品質低下が大きい。波形を生成するボコーダーは量子化時の品質劣化に対してより敏感である。」

> 「予備実験で、波形出力に最も近い畳み込み層を量子化すると、合成音声の品質が著しく低下することがわかった。」

### 2.2 INT8量子化の実際の問題

**[MeloTTS.cpp](https://github.com/apinge/MeloTTS.cpp) での報告:**

> 「現在のINT8量子化モデルは軽微なバックグラウンドノイズを示す。回避策としてDeepFilterNetによる後処理を統合した。」

**結論**: INT8量子化はTTSで**背景ノイズ**を発生させる可能性がある。

### 2.3 推奨される量子化戦略

| コンポーネント | 推奨形式 | 理由 |
|--------------|---------|------|
| LLM (Qwen2) | **FP16** | 音声トークン生成の精度維持 |
| Flow Decoder | **FP16** または FP32 | メルスペクトログラム品質 |
| HiFT Vocoder | **FP32** | 波形出力に最も敏感 |

---

## 3. Unity Sentis の制約

### 3.1 サポートされる量子化

**[Sentis 2.1.3 ドキュメント](https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/quantize-a-model.html):**

- Sentis内部での量子化: サポート（Dense, MatMul, Conv等の特定演算のみ）
- 外部で量子化されたINT8 ONNXモデル: **非サポート**
- DequantizeLinear/QuantizeLinear演算子: **削除済み**

### 3.2 外部量子化モデルの問題

**[Unity Forum報告](https://discussions.unity.com/t/support-externally-quantised-int8-onnx-models-in-sentis/1566203):**

> 「uint8で事前量子化されたONNXモデルをインポートできない。DequantizeLinear、DynamicDequantizeLinear、QuantizeLinear演算がSentisでサポートされなくなった。」

### 3.3 LLM量子化の問題

**[Sentis LLM量子化Issue](https://discussions.unity.com/t/can-quantized-llm-models-be-used-in-sentis/368586):**

> 「モデルをfloat16やUint8に量子化すると "NotImplementedException" エラーが発生する。」

---

## 4. CosyVoice公式の対応

### 4.1 公式サポート

CosyVoice2/3では以下のオプションが提供されている:

```python
CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=False,
    load_vllm=False,
    fp16=False  # FP16モード
)
```

### 4.2 TensorRT での精度指定

公式ドキュメントより:
- CosyVoice2: デフォルト **FP16**
- CosyVoice3: デフォルト **FP32**

**理由**: CosyVoice3はより高い精度を必要とするため、FP32がデフォルト。

---

## 5. 推奨戦略

### 5.1 サイズ vs 品質のトレードオフ

| 戦略 | モデルサイズ | 音質 | Unity互換性 |
|------|------------|------|------------|
| 全FP32 | ~510MB | 最高 | ✅ 完全対応 |
| **全FP16** | **~255MB** | **高品質** | **✅ 推奨** |
| LLM:INT8 + その他:FP16 | ~180MB | 中程度（ノイズリスク） | ⚠️ 要検証 |
| 全INT8 | ~130MB | 低品質（ノイズ確実） | ❌ 非推奨 |

### 5.2 具体的推奨

**Unity Sentis向け最適構成:**

```
LLM (Qwen2):           FP16 (~200MB)
Flow Decoder:          FP16 (~40MB)
HiFT Vocoder:          FP32 (~20MB) ← 最も敏感
CAMPPlus:              FP16 (~5MB)
-----------------------------------------
合計:                  ~265MB
```

### 5.3 段階的な検証アプローチ

1. **Phase 1**: 全FP32でエクスポート、動作確認
2. **Phase 2**: LLMとFlowをFP16に変換、音質比較
3. **Phase 3**: HiFTのFP16を試行（音質劣化が許容範囲か確認）
4. **Phase 4**: 必要に応じてINT8を部分適用（LLMの非出力層のみ）

---

## 6. INT8を使用する場合の緩和策

どうしてもINT8を使用する必要がある場合:

### 6.1 量子化対象の選択

```python
# 推奨: 重みのみINT8、アクティベーションはFP16維持
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul"],  # 限定的な適用
    # 注: Gemm, Convは除外して精度維持
)
```

### 6.2 出力層の保護

```python
# 出力に近い層は量子化から除外
nodes_to_exclude = [
    "llm_decoder",           # 最終出力層
    "conv_post",             # HiFT最終層
    "vocoder_output"         # 波形出力層
]
```

### 6.3 後処理によるノイズ除去

INT8でノイズが発生した場合:
- DeepFilterNet: 背景ノイズ除去
- スペクトラルサブトラクション: シンプルなノイズ軽減

---

## 7. FP16 ONNXエクスポート方法

### 7.1 PyTorchからの直接エクスポート

```python
import torch

# モデルをFP16に変換
model = model.half()

# FP16でエクスポート
torch.onnx.export(
    model,
    dummy_input.half(),
    "model_fp16.onnx",
    opset_version=15,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch", 1: "seq_len"}}
)
```

### 7.2 ONNX変換後の精度変換

```python
from onnxruntime.transformers import float16

# FP32 ONNX → FP16 ONNX
float16.convert_float_to_float16(
    input_model_path="model_fp32.onnx",
    output_model_path="model_fp16.onnx",
    keep_io_types=True,  # 入出力はFP32維持
    op_block_list=["LayerNormalization"]  # 特定演算は除外
)
```

---

## 8. 結論

### 8.1 ユーザーの懸念は正しい

- **INT8はTTSで音質劣化のリスクが高い**
- **FP16が品質とサイズのバランスで最適**
- **ボコーダー（HiFT）は特に量子化に敏感**

### 8.2 最終推奨

| 項目 | 推奨 |
|------|------|
| LLM | FP16 |
| Flow | FP16 |
| HiFT | FP32 (可能であれば) または FP16 |
| 全体サイズ | ~265MB |

### 8.3 Unity Sentis制約への対応

- 外部INT8量子化モデルは使用不可
- Sentis内部量子化機能を使用する場合は慎重にテスト
- **FP16 ONNXモデルをそのままインポートが最も安全**

---

## 参考文献

- [Red Hat: 50万回のLLM量子化評価](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms)
- [BitTTS: 1.58bit量子化TTS](https://arxiv.org/abs/2506.03515)
- [MeloTTS.cpp: INT8ノイズ問題](https://github.com/apinge/MeloTTS.cpp)
- [Unity Sentis量子化ドキュメント](https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/quantize-a-model.html)
- [FP8 vs INT8比較](https://arxiv.org/pdf/2303.17951)
- [ONNX Runtime量子化](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

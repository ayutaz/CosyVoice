# CosyVoice3 Unity Sentis互換性調査

**作成日**: 2026-01-14
**ブランチ**: feature/unity-sentis-onnx

---

## 1. 調査結果サマリー

### 1.1 現在のONNXモデルの問題点

| モデル | 現在のopset | Sentis対応 | 対応必要 |
|--------|------------|------------|---------|
| flow.decoder.estimator.fp16.onnx | **18** | 7-15のみ | **再エクスポート必要** |
| hift_source_generator_fp32.onnx | **17** | 7-15のみ | **再エクスポート必要** |
| text_embedding_fp32.onnx | **17** | 7-15のみ | **再エクスポート必要** |
| flow_pre_lookahead_fp16.onnx | 15 | OK | - |
| flow_speaker_projection_fp16.onnx | 15 | OK | - |
| flow_token_embedding_fp16.onnx | 15 | OK | - |
| hift_decoder_fp32.onnx | 15 | OK | - |
| hift_f0_predictor_fp32.onnx | 15 | OK | - |
| llm_backbone_decode_fp16.onnx | 15 | OK | - |
| llm_backbone_initial_fp16.onnx | 15 | OK | - |
| llm_decoder_fp16.onnx | 15 | OK | - |
| llm_speech_embedding_fp16.onnx | 15 | OK | - |

### 1.2 Unity Sentis制約

| 制約 | 詳細 |
|------|------|
| **Opset version** | 7-15のみサポート（推奨: 15） |
| **テンソル次元** | 最大8次元 |
| **未サポートオペレーター** | If, Log1p, FFT, IFFT, RFFT, IRFFT, GRU, RNN, Loop等 |

---

## 2. 使用オペレーター分析

### 2.1 全モデルで使用されているオペレーター

```
# 基本演算（すべてサポート）
Add, Sub, Mul, Div, Pow, Abs, Neg, Sqrt, Exp, Sin, Cos, Tanh, Sigmoid

# 活性化関数（すべてサポート）
Softmax, Elu, LeakyRelu, Softplus

# 畳み込み（サポート）
Conv

# 正規化（LayerNormalizationはサポート）
LayerNormalization

# テンソル操作（すべてサポート）
Reshape, Transpose, Concat, Split, Gather, Slice, Unsqueeze, Squeeze, Pad, Tile, Expand

# 行列演算（サポート）
MatMul, Gemm

# 削減演算（サポート）
ReduceMean

# 特殊オペレーター
Einsum       - サポート（CPUは全機能、GPUは1-2入力のみ）
ScatterND    - サポート
CumSum       - サポート
Reciprocal   - サポート
Resize       - サポート
```

### 2.2 潜在的問題オペレーター

| オペレーター | モデル | Sentisサポート | 対応 |
|-------------|--------|---------------|------|
| Einsum | flow.decoder.estimator | 部分的（GPU制限あり） | CPUフォールバックで動作 |
| ScatterND | hift_source_generator, llm_backbone | サポート | OK |
| CumSum | hift_source_generator | サポート | OK |

---

## 3. 再エクスポート計画

### 3.1 opset 15で再エクスポートが必要なモデル

1. **flow.decoder.estimator** (opset 18 → 15)
   - DiTモデル
   - Einsumオペレーターを使用

2. **hift_source_generator** (opset 17 → 15)
   - Source信号生成
   - CumSum, ScatterNDを使用

3. **text_embedding** (opset 17 → 15)
   - Qwen2のembedding層
   - Gatherのみ使用（シンプル）

### 3.2 エクスポート方針

```python
# opset_version=15を指定
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    opset_version=15,  # Unity Sentis互換
    do_constant_folding=True,  # 最適化
    ...
)
```

---

## 4. If演算子の回避（ZipVoice参考）

ZipVoiceでは、Positional Encodingの動的拡張でIf演算子が生成される問題があった。

### 4.1 解決方法

```python
# エクスポート前にPEを事前計算
def _precompute_positional_encodings(model, max_len):
    dummy_input = torch.zeros(max_len)
    for name, module in model.named_modules():
        if hasattr(module, 'extend_pe') and hasattr(module, 'pe'):
            module.extend_pe(dummy_input)

# エクスポート時に条件分岐を回避
if torch.jit.is_scripting() or torch.jit.is_tracing():
    pe = self.pe  # 事前計算済みを使用
else:
    self.extend_pe(x)  # 動的拡張
```

### 4.2 CosyVoice3での確認事項

- [ ] LLMモデルにIf演算子が含まれるか確認
- [ ] Rotary Position Embedding（RoPE）の実装確認
- [ ] 条件分岐がある場合は静的グラフに変換

---

## 5. 互換性検証スクリプト

```python
SENTIS_UNSUPPORTED_OPS = {
    "If", "Log1p", "LogAddExp", "ComplexAbs",
    "FFT", "IFFT", "RFFT", "IRFFT",
    "Unique", "StringNormalizer", "TfIdfVectorizer", "Tokenizer",
    "GRU", "RNN", "Loop",
}

def verify_sentis_compatibility(model_path):
    model = onnx.load(model_path)
    warnings = []

    # 未サポートオペレーターの検出
    for node in model.graph.node:
        if node.op_type in SENTIS_UNSUPPORTED_OPS:
            warnings.append(f"Unsupported: {node.op_type}")

    # テンソル次元の確認（8次元制限）
    for value_info in model.graph.value_info:
        if value_info.type.HasField("tensor_type"):
            dims = len(value_info.type.tensor_type.shape.dim)
            if dims > 8:
                warnings.append(f"Tensor {value_info.name}: {dims} dims > 8")

    return warnings
```

---

## 6. 実装ロードマップ

### Phase 1: 再エクスポート
- [ ] opset 15でflow.decoder.estimatorを再エクスポート
- [ ] opset 15でhift_source_generatorを再エクスポート
- [ ] opset 15でtext_embeddingを再エクスポート
- [ ] 全モデルの互換性検証

### Phase 2: Unity側実装
- [ ] ISTFT実装（n_fft=16, hop_length=4）
- [ ] Qwen2トークナイザーのC#実装
- [ ] KVキャッシュ管理
- [ ] Euler Solver（Flow用）

### Phase 3: 統合テスト
- [ ] Unityでのモデル読み込み確認
- [ ] 推論パイプライン実装
- [ ] 音声生成テスト

---

## 7. 参考資料

### Unity Sentis ドキュメント
- [Supported ONNX operators](https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/supported-operators.html)
- [Unity AI Inference Engine 2.4](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.4/manual/index.html)

### ZipVoice参考実装
- `C:\Users\yuta\Desktop\Private\ZipVoice\zipvoice\bin\onnx_export_sentis.py`
- `C:\Users\yuta\Desktop\Private\ZipVoice\CLAUDE.md`

### 既存ドキュメント
- `docs/unity-complete-onnx-strategy.md`
- `docs/onnx-export-implementation.md`

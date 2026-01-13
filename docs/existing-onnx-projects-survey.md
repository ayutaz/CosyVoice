# CosyVoice ONNX エクスポート既存プロジェクト調査

## 調査日: 2026-01-13

本ドキュメントはCosyVoice/CosyVoice2/CosyVoice3をONNXにエクスポートする既存のプロジェクト・手法についての調査結果をまとめたものです。

---

## 1. 公式リポジトリのONNXサポート

### 1.1 公式エクスポートスクリプト

**場所:** `cosyvoice/bin/export_onnx.py`

**エクスポート対象:**
- `flow.decoder.estimator` のみ
- **LLM、Speech Tokenizer、HiFT (HiFi-GAN) は未対応**

**使用方法:**
```bash
python cosyvoice/bin/export_onnx.py --model_dir pretrained_models/CosyVoice2-0.5B
```

**出力ファイル:**
```
{model_dir}/flow.decoder.estimator.fp32.onnx
```

**技術仕様:**
- Opset: 18
- 入力: `x`, `mask`, `mu`, `t`, `spks`, `cond`
- 出力: `estimator_out`
- 動的軸: `seq_len`（シーケンス長）

### 1.2 ロードオプション

CosyVoice2初期化時のパラメータ:
```python
CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=True,    # JITコンパイル
    load_onnx=False,  # ONNX使用
    load_trt=False    # TensorRT使用
)
```

### 1.3 既知の問題

**Issue #192: LLM → ONNXエクスポート未対応**
- [GitHub Issue #192](https://github.com/FunAudioLLM/CosyVoice/issues/192)
- 状況: LLMのONNXエクスポートコードは**未公開**
- 公式回答: 「プロダクションではlibtorchにエクスポートしているが、コードは未公開」
- LLMエンコーダは処理時間の約82%を占める

**Issue #1030: ONNXエクスポート時の検証エラー**
- [GitHub Issue #1030](https://github.com/FunAudioLLM/CosyVoice/issues/1030)
- エラー内容: 数値不一致 (4.5%)、許容誤差を超過
- ワークアラウンド: 検証チェックをコメントアウトすると動作する
- 実用上は問題なく音声生成可能との報告

---

## 2. 関連プロジェクト

### 2.1 S3Tokenizer

**リポジトリ:** [xingchensong/S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

**概要:**
- CosyVoiceの音声トークナイザー（S3Tokenizer）のPyTorch再実装
- 公式がONNXのみ公開のため、リバースエンジニアリングで作成

**ONNX関連機能:**
- `utils.py::onnx2torch()`: ONNX重みをPyTorchに変換
- 公式ONNXファイルから重みを抽出してPyTorchモデルに読み込み
- バッチ処理対応（元のONNXはバッチサイズ1のみ）
- 約790倍の高速化を実現

**サポートバージョン:**
- S3Tokenizer V1 (50Hz, 25Hz)
- S3Tokenizer V2 (25Hz)
- S3Tokenizer V3 (25Hz)

### 2.2 LightTTS

**リポジトリ:** [ModelTC/LightTTS](https://github.com/ModelTC/LightTTS)

**概要:**
- CosyVoice2/3向けの軽量高速推論フレームワーク
- LightLLMベースのLLM高速化

**サポート内容:**
- CosyVoice2、CosyVoice3両対応
- TensorRT統合（`load_trt=True`推奨）
- JITコンパイル（`load_jit=True`）
- ストリーミング・双方向ストリーミング

**ONNX対応:** 明示的なONNXエクスポート機能なし（TensorRT/JITに注力）

**パフォーマンス (RTX 5090):**
- 1.73 QPS（非ストリーミング）
- RTF 0.11（リアルタイムファクター）

### 2.3 sherpa-onnx

**リポジトリ:** [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)

**概要:**
- 多言語対応の音声認識・合成フレームワーク
- ONNX Runtimeベース、オフライン動作
- 12プログラミング言語対応

**CosyVoice対応:**
- **直接サポートなし**
- CosyVoiceはSiliconFlow API経由での利用のみ記載
- 対応TTSモデル: VITS、MeloTTS等

**プラットフォーム:**
- Android, iOS, HarmonyOS
- Raspberry Pi, RISC-V
- x86_64サーバー
- WebSocket

---

## 3. コンポーネント別エクスポート状況

| コンポーネント | 公式ONNX | サードパーティ | 備考 |
|--------------|---------|--------------|------|
| Speech Tokenizer | ✅ 公開済み | S3Tokenizer (PyTorch変換) | ONNX→PyTorch変換可能 |
| LLM (Qwen系) | ❌ 未公開 | なし | プロダクションではlibtorch使用 |
| Flow Decoder Estimator | ✅ `export_onnx.py` | なし | 公式スクリプトで出力可能 |
| HiFT (HiFi-GAN) | ❌ 未対応 | なし | 追加実装が必要 |
| CAMPPlus (話者埋め込み) | ✅ 公開済み | なし | campplus.onnx |

---

## 4. Unity/C#対応状況

**調査結果: 既存のUnity対応プロジェクトは存在しない**

### 可能なアプローチ

1. **サーバーサイド推論**
   - CosyVoiceをPython/gRPCサーバーとして実行
   - Unityからネットワーク経由で呼び出し
   - 公式Docker対応あり

2. **部分的ONNXエクスポート**
   - Flow Decoder Estimator: 公式対応済み
   - HiFT: 追加実装が必要
   - LLM: 最大の課題（未公開）

3. **代替アーキテクチャ**
   - ZipVoice（Flow Matching TTS）: 完全なONNXサポート
   - Piper TTS (VITS): Unity対応実績あり

---

## 5. 技術的課題と解決策

### 5.1 LLM ONNXエクスポートの課題

**問題:**
- Qwen系LLMは~400Mパラメータと巨大
- 自己回帰生成はONNXでの表現が複雑
- KVキャッシュの動的サイズ管理

**可能な解決策:**
1. **モデル蒸留**: 小型モデルを学習
2. **量子化**: INT8/INT4でサイズ削減
3. **ハイブリッド構成**: LLMはサーバー、その他はローカル

### 5.2 Flow Decoder Estimatorの課題

**Issue #1030の問題:**
- 数値不一致（4.5%のミスマッチ）
- 原因: 浮動小数点精度の差異

**解決策:**
- 許容誤差を緩和（`rtol=1e-2, atol=1e-4` → 調整）
- 実用上は問題なく動作するとの報告

### 5.3 HiFTエクスポート

**状況:** 公式スクリプトに含まれていない

**実装アプローチ:**
```python
# HiFTエクスポート（未実装・要追加）
hift = model.model.hift
torch.onnx.export(
    hift,
    (mel_input,),
    'hift.onnx',
    input_names=['mel'],
    output_names=['audio'],
    dynamic_axes={'mel': {2: 'time'}, 'audio': {1: 'samples'}}
)
```

---

## 6. 推奨アプローチ

### 短期目標（検証フェーズ）

1. **公式Flow Decoderエクスポートの検証**
   ```bash
   python cosyvoice/bin/export_onnx.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
   ```

2. **HiFTのONNXエクスポート追加**
   - `cosyvoice/bin/export_onnx.py`を拡張

3. **Speech TokenizerのONNX取得**
   - 公式モデルディレクトリに含まれる場合あり
   - または S3Tokenizer プロジェクトを参照

### 中期目標

1. **LLMの代替検討**
   - サーバーハイブリッド構成
   - または小型蒸留モデル

2. **Unity統合プロトタイプ**
   - uZipVoiceのアーキテクチャを参考に実装

---

## 7. 参考リンク

### 公式リソース
- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - メインリポジトリ
- [Issue #192: LLM ONNX Export](https://github.com/FunAudioLLM/CosyVoice/issues/192) - LLMエクスポートの議論
- [Issue #1030: ONNX Export Error](https://github.com/FunAudioLLM/CosyVoice/issues/1030) - エクスポートエラー

### 関連プロジェクト
- [xingchensong/S3Tokenizer](https://github.com/xingchensong/S3Tokenizer) - Speech Tokenizer PyTorch実装
- [ModelTC/LightTTS](https://github.com/ModelTC/LightTTS) - 軽量推論フレームワーク
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) - 多言語音声処理ツールキット

### ドキュメント
- [sherpa-onnx TTS Documentation](https://k2-fsa.github.io/sherpa/onnx/tts/index.html)
- [CosyVoice 3.0 Tech Guide](https://stable-learn.com/en/cosyvoice3-tech-guide/)

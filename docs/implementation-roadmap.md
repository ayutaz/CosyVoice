# CosyVoice3 Unity ONNX実装ロードマップ

## 概要

**目標**: CosyVoice3をFP16 ONNXでエクスポートし、Unity Sentisで完全オフライン動作を実現

**精度方針**: FP16（HiFTボコーダーはFP32推奨）

**推定モデルサイズ**: ~265MB

---

## Phase 1: 基盤検証（1週目）

### 1.1 環境セットアップ

- [ ] Python環境でのONNXエクスポート依存関係インストール
  ```bash
  pip install onnx onnxruntime onnxruntime-gpu optimum transformers
  ```
- [ ] Unity 6プロジェクト作成
- [ ] Unity Sentis (com.unity.ai.inference) パッケージインストール

### 1.2 HiFT (ボコーダー) ONNXエクスポート検証

**優先度: 最高** - 最も単純で検証しやすい

**タスク:**
- [ ] HiFTGenerator の入出力仕様確認
- [ ] STFT/ISTFT分離のラッパークラス作成
- [ ] FP32でONNXエクスポート
- [ ] ONNX Runtime (Python) での推論検証
- [ ] Unity Sentisでのインポート・推論検証

**検証ポイント:**
- 波形出力の品質（PyTorch vs ONNX比較）
- Unity Sentisでの互換性

**成果物:**
- `scripts/export_hift_onnx.py`
- `models/hift_fp32.onnx`

### 1.3 Flow Decoder ONNXエクスポート検証

**優先度: 高**

**タスク:**
- [ ] 公式 `export_onnx.py` の実行
- [ ] FP16変換の適用
- [ ] Euler Solverの動作確認（Python）
- [ ] Unity Sentisでのインポート検証

**検証ポイント:**
- メルスペクトログラム出力の品質
- 10ステップEuler統合の精度

**成果物:**
- `models/flow_decoder_fp16.onnx`

---

## Phase 2: ボコーダー統合（2週目）

### 2.1 Unity C#基盤実装

**タスク:**
- [ ] プロジェクト構造作成
  ```
  Assets/uCosyVoice/
  ├── Runtime/
  │   ├── Core/
  │   ├── Inference/
  │   ├── Audio/
  │   └── Utils/
  ├── Models/
  └── Tests/
  ```
- [ ] Tensor操作ユーティリティ実装
- [ ] AudioClipBuilder実装

### 2.2 HiFT Unity実装

**タスク:**
- [ ] MiniISTFT (n_fft=16) C#実装
- [ ] SineGenerator (F0→正弦波) C#実装
- [ ] HiFTInference クラス実装
- [ ] 単体テスト作成

**検証ポイント:**
- C# ISTFT出力とPython出力の一致
- AudioClip再生品質

### 2.3 Flow Decoder Unity実装

**タスク:**
- [ ] EulerSolver C#実装（uZipVoiceから移植）
- [ ] FlowDecoderInference クラス実装
- [ ] HiFTとの結合テスト

**検証ポイント:**
- 音声トークン（ダミー）→ 波形の生成
- 音質確認

---

## Phase 3: LLMエクスポート検証（3週目）

### 3.1 Qwen2 ONNXエクスポート調査

**タスク:**
- [ ] CosyVoice3LMのアーキテクチャ詳細分析
- [ ] HuggingFace Optimumでのエクスポート試行
- [ ] カスタムラッパーでのエクスポート試行
- [ ] KVキャッシュ構造の確認

**検証ポイント:**
- ONNXエクスポート成功可否
- FP16変換後のサイズ
- Unity Sentisインポート可否

### 3.2 LLM分割戦略検討

**課題**: LLMが大きすぎる場合の対策

**オプション:**
1. 単一ONNXファイル（~200MB FP16）
2. レイヤー分割（複数ONNX）
3. 埋め込み層分離

**タスク:**
- [ ] メモリ使用量測定
- [ ] 推論時間測定
- [ ] 最適な分割戦略決定

---

## Phase 4: LLM Unity統合（4週目）

### 4.1 KVキャッシュ管理実装

**タスク:**
- [ ] KVCacheManager C#実装
- [ ] スライディングウィンドウ方式実装
- [ ] メモリ効率化

### 4.2 LLM推論実装

**タスク:**
- [ ] LLMInference クラス実装
- [ ] 自己回帰ループ実装
- [ ] サンプリング戦略実装（Top-K）

**検証ポイント:**
- 音声トークン生成の品質
- 推論速度

### 4.3 話者エンコーダ統合

**タスク:**
- [ ] CAMPPlus ONNXの取得/検証
- [ ] SpeakerEncoderInference 実装
- [ ] プロンプト音声からの埋め込み抽出

---

## Phase 5: 統合テスト（5週目）

### 5.1 フルパイプライン統合

**タスク:**
- [ ] CosyVoiceManager 統合クラス実装
- [ ] テキスト→音声のE2Eテスト
- [ ] 日本語テキストでの検証

### 5.2 パフォーマンス最適化

**タスク:**
- [ ] 推論時間計測・ボトルネック特定
- [ ] メモリ使用量最適化
- [ ] バッチ処理検討

### 5.3 品質評価

**タスク:**
- [ ] PyTorch出力との比較
- [ ] 主観的音質評価
- [ ] 問題点の洗い出し

---

## Phase 6: 最終調整（6週目）

### 6.1 最適化適用

**タスク:**
- [ ] 必要に応じてFP16/FP32調整
- [ ] Sentisバックエンド最適化（GPUPixel等）
- [ ] メモリリーク修正

### 6.2 ドキュメント整備

**タスク:**
- [ ] API ドキュメント作成
- [ ] 使用方法ガイド作成
- [ ] トラブルシューティングガイド

### 6.3 サンプル作成

**タスク:**
- [ ] デモシーン作成
- [ ] サンプルスクリプト作成

---

## 検証チェックリスト

### ONNXエクスポート検証

| コンポーネント | エクスポート | Python推論 | Unity推論 | 品質確認 |
|--------------|------------|-----------|----------|---------|
| HiFT (FP32) | ⬜ | ⬜ | ⬜ | ⬜ |
| Flow Decoder (FP16) | ⬜ | ⬜ | ⬜ | ⬜ |
| LLM (FP16) | ⬜ | ⬜ | ⬜ | ⬜ |
| CAMPPlus (FP16) | ⬜ | ⬜ | ⬜ | ⬜ |

### Unity実装検証

| 機能 | 実装 | 単体テスト | 統合テスト |
|-----|------|----------|----------|
| MiniISTFT | ⬜ | ⬜ | ⬜ |
| EulerSolver | ⬜ | ⬜ | ⬜ |
| KVCacheManager | ⬜ | ⬜ | ⬜ |
| AudioClipBuilder | ⬜ | ⬜ | ⬜ |

### E2E検証

| テストケース | 結果 |
|------------|------|
| 短い日本語テキスト（5文字） | ⬜ |
| 中程度の日本語テキスト（20文字） | ⬜ |
| 長い日本語テキスト（100文字） | ⬜ |
| プロンプト音声変更 | ⬜ |

---

## リスクと緩和策

| リスク | 影響度 | 緩和策 |
|-------|-------|-------|
| LLM ONNXエクスポート失敗 | 高 | カスタムラッパー作成、レイヤー分割 |
| Unity Sentis互換性問題 | 高 | Opset調整、演算子置換 |
| メモリ不足 | 中 | ストリーミング推論、チャンク処理 |
| 推論速度不足 | 中 | GPUバックエンド、最適化 |
| 音質劣化 | 中 | FP32フォールバック、後処理 |

---

## 成果物一覧

### Python スクリプト

```
scripts/
├── export_hift_onnx.py          # HiFTエクスポート
├── export_flow_onnx.py          # Flowエクスポート
├── export_llm_onnx.py           # LLMエクスポート
├── convert_fp16.py              # FP16変換
└── verify_onnx.py               # ONNX検証
```

### Unity パッケージ

```
Assets/uCosyVoice/
├── Runtime/
│   ├── Core/
│   │   └── CosyVoiceManager.cs
│   ├── Inference/
│   │   ├── HiFTInference.cs
│   │   ├── FlowDecoderInference.cs
│   │   ├── LLMInference.cs
│   │   └── SpeakerEncoderInference.cs
│   ├── Audio/
│   │   ├── MiniISTFT.cs
│   │   ├── SineGenerator.cs
│   │   └── AudioClipBuilder.cs
│   └── Utils/
│       ├── EulerSolver.cs
│       └── KVCacheManager.cs
├── Models/
│   ├── hift_fp32.onnx
│   ├── flow_decoder_fp16.onnx
│   ├── llm_fp16.onnx
│   └── campplus_fp16.onnx
└── Tests/
```

---

## 次のアクション

**即座に開始:**
1. HiFT ONNXエクスポートスクリプト作成
2. Unity 6プロジェクト作成
3. 基本的なC#ユーティリティ実装

**今週中:**
- Phase 1.2 (HiFT検証) 完了
- Phase 1.3 (Flow検証) 開始

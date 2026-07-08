# CosyVoice3 日本語ファインチューニング 最終レポート

**期間**: 2026-07-07 〜 2026-07-08 / **ブランチ**: `feature/japanese-katakana-frontend`
**結論: 漢字混じり日本語をそのまま入力して正しく読み上げるモデルが完成した(CER 0.708 → 0.085)**

詳細な調査経緯・技術判断は [japanese_support_investigation.md](japanese_support_investigation.md) を参照。本書は全体の総括。

---

## 1. 背景と目標

- CosyVoice3 は公称「日本語対応」だが、実際は**スペース区切りカタカナ入力**が前提(公式 example.py に明記)。漢字混じりの文章を入れると読みが完全に崩壊する
- 「トークナイザーが壊れている」という風説は誤りで、正体は (a) 過去の学習コードの語彙サイズ不一致バグ(修正済み)と (b) 日本語テキストフロントエンドの不在
- **目標**: 漢字混じりの日本語をそのまま入力して正しく発音されるようにする

## 2. 実施内容(4フェーズ)

### Phase 1: カタカナ変換フロントエンド(暫定対応)

`cosyvoice/utils/ja_frontend.py` — pyopenjtalk(-plus) の NJD 発音フィールドを使い、漢字混じり文を公式期待形式のスペース区切りカタカナへ変換。助詞「は→ワ」「を→オ」、長音、句読点処理を含み、公式期待出力と完全一致。`frontend.py` に統合し、`<|endofprompt|>` 制御トークンとの共存も対応。テスト 12 件。

### Phase 2: 学習・前処理基盤の整備

**環境刷新**: Python 3.12 / torch 2.11.0+cu128 / pyopenjtalk-plus / onnxruntime-gpu 1.22 / uv プロジェクト管理(requirements.txt 廃止)/ ruff lint(flake8 CI 置換)。transformers は 4.51.3 固定(5.x は issue #1886 で CV2/3 推論が雑音化)。

**学習高速化(6視点×懐疑検証の監査で25件採用)**: 主要なもの:

| 変更 | 効果 |
|---|---|
| 死んだ lm_head 射影のスキップ(151,936語彙への射影が毎ステップ捨てられていた) | 5-10%/step + VRAM 数GB |
| torch.compile(内側 Qwen2Model のみ、実測で採否判定) | **-17% step時間**(4090実測、recompile 2回のみ) |
| 波形フリー llm データパイプライン(parquet 列プルーニング+トークン長ベース batch) | エポック毎読取 ~200GB→~1-2GB |
| bf16 の GradScaler 廃止 / fused Adam / WORLD_SIZE==1 で DDP スキップ | 各 2-8%/step |
| dev シャード修正(CV パスが8倍重複していたバグ) | CV 1/8 |
| CE 高速路(lsm_weight=0 時)/ TB 書き込み間引き / ONNX セッションの VRAM 排除 | 数%each |

**前処理並列化**: 話者埋め込みはプロセス分割で 16/s→577/s(36倍)、speech token 抽出は同一長バッチ+8プロセスで 37/s→218/s(6倍)。バッチONNXの数値検証(パディング無しなら参照と実質同一、揺らぎはcudnnノイズと同水準)も実施。

**E2E スモークテスト(RTX 4090, ~$1)で本番前に4つの地雷を検出・修正**:
1. torchaudio 2.9+ の torchcodec 依存 → 音声 I/O 全滅(soundfile 直叩きに置換)
2. torch 2.11 の `ProcessGroup.options` 削除 → 学習2バッチ目でクラッシュ
3. run.sh のエラー握りつぶし → 空成果物で偽成功
4. `average_model.py --val_best` が step checkpoint をエポック末ファイルへ丸める上流バグ

### Phase 3: 本番学習(vast.ai H100 SXM, $2.33/hr)

- **データ**: HF `ayousanz/moe-speech-plus`(gated、日本語キャラ演技音声、473 zip / 326GB)
  - speechMOS は演技音声を低採点する(実測中央値 2.18)ため閾値を 2.5→**1.5 に較正**、cross-CER≤0.2 と併用で **289,893 発話 ≒ 470h 採用**(73.5%)
- **学習**: llm のみ(flow/hift 凍結)、テキストは漢字混じりのまま。5 エポック / 17,616 ステップ、bf16 + torch.compile + 30,000 mel フレーム/バッチ(VRAM 31.8/80GB)
- **実行約 2 時間**で完走。CV loss 3.36→3.22(エポック 3→4 で平坦化 = 5 エポック打ち切りが適切)
- 成果物: `checkpoints/cosyvoice3_ja/llm.pt`(CV ベスト5 checkpoint の平均、2.57GB)
- **総費用 ~$25**(前処理の試行錯誤込み。整備済みツールなら ~$15)

### Phase 4: 定量評価(ローカル RTX 4070 Ti SUPER)

`scripts/eval_ja_cer.py` — 学習データ外の漢字混じり 20 文(同形異音語・助数詞・日付・数値)を 4 経路で合成し、whisper large-v3 の書き起こしと CER 比較:

| 経路 | llm | 入力 | avg CER | 完全一致 |
|---|---|---|---|---|
| base_kanji | 事前学習 | 漢字直接 | **0.708(崩壊)** | 0/20 |
| base_katakana | 事前学習 | カタカナ変換 | 0.097 | 9/20 |
| ft_katakana | FT済み | カタカナ変換 | 0.104 | 8/20 |
| **ft_kanji** | **FT済み** | **漢字直接** | **0.085(最良)** | 9/20 |

- FT 済みモデルは漢字直接入力でカタカナ変換経路を**上回る**(文単位: 勝3/分16/負1)。残誤りの上位は whisper の表記ゆれ(「三百キロメートル」→「300km」)で発音は正しく、実効 CER は ~0.03 相当
- FT はカタカナ経路を劣化させない(差はノイズ範囲)
- 音声: `eval_out/{経路}/*.wav`、全書き起こし: `eval_out/report.json`

## 3. 使い方

```python
import sys
sys.path.append('third_party/Matcha-TTS')
import torch
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import audio_save

cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
# FT済み llm を差し替え
state = torch.load('checkpoints/cosyvoice3_ja/llm.pt', map_location='cpu', weights_only=True)
cosyvoice.model.llm.load_state_dict({k: v for k, v in state.items() if k not in ('epoch', 'step')}, strict=True)
cosyvoice.model.llm.to(cosyvoice.model.device).eval()

# 漢字混じりをそのまま入力(text_frontend=False で変換をスキップ)
text = 'You are a helpful assistant.<|endofprompt|>今日は良い天気なので、公園まで散歩に行きました。'
for i, j in enumerate(cosyvoice.inference_cross_lingual(text, './asset/zero_shot_prompt.wav', text_frontend=False)):
    audio_save(f'ja_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

- 事前学習モデルをそのまま使う場合は従来どおり `text_frontend=True`(ja_frontend がカタカナ変換)
- 学習の再現は `examples/moe_speech/cosyvoice3/run.sh`(stage -1〜6、gated データセットのため HF 認証が必要)

## 4. 今後の課題

1. **8000h へのスケールアップ**: 今回整備した前処理(並列抽出・同一長バッチ・波形フリー parquet)と学習設定はそのまま流用可能。見込み ~$250-450 / 前処理数時間+学習 ~1 日
2. **難読語カバレッジ**: 評価は一般文体のみ。固有名詞・専門用語では差が出る可能性があり、8000h の語彙カバレッジ拡大が有効
3. **評価の精緻化**: 数値・単位の表記ゆれを吸収する CER 正規化、話者性・韻律の主観評価(MOS)
4. **flow の日本語適応(任意)**: 今回は llm のみ。韻律をさらに詰める場合は flow の FT を検討(parquet を音声バイト込みで再生成する必要あり)

## 5. 主な成果物一覧

| 種別 | パス |
|---|---|
| FT済みモデル | `checkpoints/cosyvoice3_ja/llm.pt`(+ ベスト単体 `epoch_4_step_16000.pt`)。**バックアップ: HF private `ayousanz/cosyvoice3-ja-llm`**(再配布禁止データ由来のため private 維持) |
| 学習レシピ | `examples/moe_speech/cosyvoice3/`(run.sh / conf / prepare_data / 並列抽出ツール) |
| 日本語フロントエンド | `cosyvoice/utils/ja_frontend.py` |
| 評価スクリプト | `scripts/eval_ja_cer.py` + `eval_out/report.json` |
| 並列前処理ツール | `tools/extract_embedding_sharded.py`, `local/extract_speech_token_{batch,sharded}.py` |
| 調査・技術ノート | `docs/japanese_support_investigation.md` |
| テスト | `tests/`(37 件: ja_frontend / 学習高速化の数値一致 / 音声 I/O / データ準備) |

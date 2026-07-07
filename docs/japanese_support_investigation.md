# CosyVoice3 日本語対応 調査報告

作成日: 2026-07-07 / 最終更新: 2026-07-07 (Phase 1 実装をブランチに反映、Phase 3 を保有データ前提に更新) / 対象: main @ 074ca6d (2026-05-26)

## TL;DR

- **トークナイザー本体は壊れていない**。CosyVoice3 は Qwen2 の BPE トークナイザーを使っており、日本語テキストのエンコード自体は問題なく行える。
- 「壊れていた」のは**学習コードの語彙サイズ不一致**(issue #1705)で、現在の main には修正済み。ただし `forward_dpo` に残骸あり(後述)。
- 本当の問題は**テキストフロントエンドに日本語処理が存在しない**こと。公式も「日本語はスペース区切りカタカナに変換して入力せよ」と `example.py` に注記しており、漢字混じり文をそのまま入れると誤読(中国語読み)が多発する。
- fork には過去の日本語対応ブランチが 2 本あるが、**どちらも肝心の漢字→カナ変換が不完全**。
- 今回 pyopenjtalk (NJD の発音フィールド) ベースの変換を検証し、**公式サンプルの期待出力と完全一致**することを確認した。実装はブランチ `feature/japanese-katakana-frontend` に反映済み(Phase 1 完了、未コミット)。
- ファインチューニング (Phase 3) は**ユーザー保有の高品質日本語データ 600 時間で先行**し、結果を見て 8,000 時間へスケールアップする方針(2026-07-07 決定)。

---

## 1. 現状の main における日本語対応

### 1.1 公称スペックと実態

- README / HF モデルカード: 9 言語対応(日本語を含む)。
- 実態: `example.py:98` に公式注記
  「`NOTE for Japanese usage, you must translate it to katakana.`」
  例: `歴史的世界においては、…` → `レキシ テキ セカイ ニ オイ テ ワ、…`(スペース区切りカタカナ、助詞「は」も発音の「ワ」に変換)。
- 漢字トークンが中国語と共有されているため、漢字混じり文は中国語読み・誤読が多発する
  ([issue #303](https://github.com/FunAudioLLM/CosyVoice/issues/303): 「晴子」「決めました」等の誤読報告。開発者回答は「BPE なので学習データ量に依存。日本語データで追加学習するか読みを変えて入力せよ」)。
- 漢字→カタカナ変換機能は**リポジトリ内に存在しない**(pyopenjtalk / MeCab 等への依存なし)。

### 1.2 トークナイザー (`cosyvoice/tokenizer/tokenizer.py`)

- `CosyVoice3Tokenizer` (L274): Qwen2 `AutoTokenizer` + 特殊トークン追加。
- 発音修正 (pronunciation inpainting) 用特殊トークンは**中国語ピンイン + 英語 CMU 音素のみ**。日本語音素・かな用のトークンは無い。
- 日本語のエンコード自体はバイトレベル BPE で問題なし。**トークナイザーは壊れていない**。

### 1.3 「壊れていた」話の正体: 学習コードの語彙サイズ不一致

- [issue #1705](https://github.com/FunAudioLLM/CosyVoice/issues/1705):
  CV3 は `llm_decoder` 出力が `speech_token_size + 200` (=6761) なのに、精度計算が CV2 用の `+3` (=6564) のままで
  `RuntimeError: shape '[-1, 6564]' is invalid` が発生 → 上流で修正済み。
- 現 main は `cosyvoice/llm/llm.py:404` で `self.llm_decoder.out_features` を使う形になっており修正済み。
- **残骸**: `llm.py:447` の `forward_dpo` は `speech_token_size + 3` 決め打ちのまま。CV3 で DPO 学習をすると同じ形状エラーになる(通常学習・推論には無関係)。

### 1.4 テキストフロントエンドの問題 (`cosyvoice/cli/frontend.py` / `frontend_utils.py`)

`text_normalize` に日本語分岐が無く、`contains_chinese` (U+4E00–9FFF) が漢字にマッチするため:

| 入力 | 通るパス | 問題 |
|---|---|---|
| 漢字を含む日本語 | 中国語パス | wetext の中国語正規化(数字が中国語読み化など)。ttsfrd 環境では `pinyinvg` (ピンイン)処理 |
| かなのみ(公式推奨のカタカナ入力含む) | 英語パス | 数字が英語読み化(inflect)。文分割が `.?!;:` のみで「。」を認識せず**長文が分割されない**([issue #152](https://github.com/FunAudioLLM/CosyVoice/issues/152)) |

- instruct 指示リスト (`cosyvoice/utils/common.py:28`) は中国語方言・感情・速度のみで日本語指示は学習リストに無い。
- `text_normalize` は `<|` を含むテキストの正規化を丸ごとスキップするため、CV3 の作法
  (`You are a helpful assistant.<|endofprompt|>` プレフィックス付き入力)では**そもそも正規化が走らない**。
  日本語フロントエンドを入れる場合はプレフィックス分離処理が必要(検証済み、§3 参照)。

---

## 2. 既存ブランチの調査 (origin = ayutaz/CosyVoice)

### 2.1 `feature/japanese-support` (2025-12-15, 1 コミット, CV3 期の main ベース)

- 内容: `cosyvoice/utils/japanese_frontend.py` (146 行) + `detect_language` (ja>zh>en 優先) +
  `split_paragraph` の "ja" モード(文字数ベース) + pyopenjtalk 依存追加 + `make_parquet_list.py` の UTF-8 修正。
- **致命的な欠落**: frontend.py から呼ばれる `JapaneseTextNormalizer.normalize()` は
  全角→半角変換・記号除去・空白整理**のみ**で、漢字→カナ変換をしない。
  `text_to_kana()` / `text_to_phoneme()` メソッドは実装されているが**どこからも呼ばれていない**。
- 評価: 誤読問題は未解決。ただし言語判定の優先順位設計・"ja" 分割モード・データ準備スクリプト修正は再利用価値あり。

### 2.2 `feature/japanese-hybrid-preprocessing` (2025-09〜10, CV2 期ベース, main から約 8 ヶ月乖離)

- 内容: `cosyvoice/utils/japanese_utils.py` — `pyopenjtalk.g2p()` の音素列を**自作テーブルで「ひらがな」に逆変換**し、`<|jp|>` タグを付与する方式。実験スクリプト多数、テスト 242 行、ドキュメント 2 本。
- 問題点:
  - 音素→かな逆変換はロッシー(長音が「きょお」式になる、無声化母音の復元など)でテーブル漏れのリスクが高い。
    公式期待形式(スペース区切り**カタカナ**+長音「ー」)とも一致しない。
  - `<|jp|>` タグは CV1 の言語タグ `<|ja|>` とも違い(タグ名が誤り)、CV2/CV3 にはそもそも言語タグ方式が無い。
  - main と 7,900 行規模の差分があり、そのままのマージは非現実的。
- **価値ある資産**(取り込み推奨):
  - `docs/japanese_improvement_guide.md` (1,046 行): Phase 1 前処理 → Phase 2 アーキテクチャ → Phase 3 データ/学習 → Phase 4 評価のロードマップ。Style-Bert-VITS2 (日本語 800h で MOS 4.37) 等の先行事例整理。
  - `docs/tokenizer_comparison.md` (556 行): pyopenjtalk-plus vs kabosu-core の詳細比較。
    結論: プロダクションは pyopenjtalk-plus、同音異義語の読み分け実験は kabosu-core (yomikata)。
    ※ pyopenjtalk-plus の wheel は Python 3.11+ 対象。本リポジトリの環境は 3.10 のため、当面は素の pyopenjtalk 0.4.1 (3.10 wheel あり) が適合。
  - `examples/libritts/cosyvoice2/conf/cosyvoice2_japanese.yaml`: CV2 用日本語 FT 設定の雛形。
- tokenizer.py への変更は無し(main との diff に見える差分は本家側の進化分)。

### 2.3 その他

- SSD (speculative decoding) 関連の実装は `feature/speech-speculative-decoding` ブランチにあり、main には未マージ。

---

## 3. 今回の検証結果 (2026-07-07)

pyopenjtalk 0.4.1 (Windows / Python 3.10 wheel) で、`run_frontend()` の NJD **発音フィールド (pron)** を直接使う方式を実装・検証した。g2p の音素経由ではないため逆変換が不要で、公式期待形式に直接一致する。

| 検証項目 | 結果 |
|---|---|
| 公式サンプル文(example.py の変換例) | **期待出力と完全一致** |
| 助詞の発音変換 | は→ワ、を→オ ✓ |
| 数詞・助数詞 | 9月12日→クガツ ジューニニチ ✓ |
| 長音 | 有→ユー、東京→トーキョー(「ー」形式)✓ |
| 英単語 | YouTube→ユーチューブ、AI→エーアイ(辞書ベース)✓ |
| issue #303 の誤読例 | 晴子→ハルコ ✓ (湘北→ショーキタは固有名詞辞書の限界) |
| アクセント記号「'」の除去 | ✓ |
| `[breath]` / `[j][ǐ]` 等の制御トークン | 変換スキップで保護 ✓ |
| `<|endofprompt|>` プレフィックス | プレフィックス保持+ペイロードのみ変換、長文分割時は各セグメントに再付与 ✓ |
| 長文分割 | 「。」「、」で分割+マージ(zh モード流用)✓ |
| ユニットテスト | 12 件パス |

**実装の所在**: ブランチ `feature/japanese-katakana-frontend` に復元済み(2026-07-07、未コミット)。
- `cosyvoice/utils/ja_frontend.py`(変換本体)
- `cosyvoice/cli/frontend.py`(`text_normalize` への統合)
- `tests/test_ja_frontend.py`(12 件パス)
- `example.py` / `pyproject.toml`(pyopenjtalk==0.4.1)

バックアップ(同内容): `<scratchpad>/ja_impl_backup/`(ja_frontend.py, test_ja_frontend.py, tracked_changes.patch)

### 既知の限界(前処理方式そのもの)

1. **アクセント情報の消失**: カタカナ化で高低アクセントは伝わらない(モデルがカタカナ列から推定)。
2. **同形異音語**: OpenJTalk の文脈解析精度に依存(「生」せい/なま 等)。改善したい場合は yomikata / kabosu-core の併用が候補。
3. **固有名詞**: 辞書に無い読みは分解読みになる(湘北→ショーキタ)。ユーザー辞書 (`pyopenjtalk.mecab_dict_index`) で対処可能。

---

## 4. 実装フェーズ提案

### Phase 1: 前処理フロントエンド — **実施済み (2026-07-07)**

ブランチ `feature/japanese-katakana-frontend` に実装済み(§3)。漢字混じり入力の誤読を学習ゼロで大幅減。
残タスク: コミット、`feature/japanese-support` ブランチの言語判定設計との突き合わせレビュー。

### Phase 2: 実音声での評価パイプライン(~1 日)

- `uv sync`(全依存は pyproject.toml / uv.lock 管理)+ `Fun-CosyVoice3-0.5B` ダウンロード。
- JSUT (basic5000) 等の文を合成 → ASR (Whisper large-v3 / Fun-ASR) で書き起こし → CER で自動評価。
- 「変換なし」vs「カタカナ変換あり」vs「FT 後(漢字直入力)」の 3 条件比較がゴール。FT の前にベースライン 2 条件を測っておく。

### Phase 3: 日本語ファインチューニング(本命)— 保有データで実施

**方針決定 (2026-07-07)**: ユーザー保有の高品質日本語データセット **600 時間**で先行実施し、
結果(CER / 話者類似度 / 主観品質)を見て **8,000 時間**へスケールアップする。
公開データ(Emilia-JA 等)は不足時の補完・評価用に格下げ。

漢字混じりテキストのまま読めるよう **LLM 部のみ**追加学習する。flow / hift は凍結で開始
(発音の学習は speech token 列を出す LLM の仕事のため。音質・話者性が不足する場合のみ flow の FT を検討)。

#### 学習パイプライン(`examples/libritts/cosyvoice3/run.sh` ベース)

| Stage | 内容 | 日本語向けの変更 |
|---|---|---|
| 0 | `wav.scp` / `text` / `utt2spk` / `spk2utt` / `instruct` 生成 | **保有データの形式に合わせた `prepare_data.py` を新規作成**(形式確認待ち)。instruct は全発話 `You are a helpful assistant.<|endofprompt|>` |
| 1 | campplus 話者埋め込み抽出 (`tools/extract_embedding.py`) | 変更不要 |
| 2 | speech token 抽出 (`tools/extract_speech_token.py` + `speech_tokenizer_v3.onnx`) | 変更不要。GPU 必須級(600h で数時間規模)。音声は 16kHz 換算で処理、30 秒超セグメントは不可 |
| 3 | parquet 化 (`tools/make_parquet_list.py`) | **要修正**: 現行はエンコーディング未指定(Windows で cp932 事故)+ `split()` が日本語スペースを破壊。`feature/japanese-support` @ a05b026 に UTF-8 + `maxsplit=1` の修正済み版があるので移植する |
| 5 | `train.py --model llm`、`llm.pt` から初期化 (torch_ddp / bf16 AMP) | 変更不要(ハイパラは要調整) |
| 6 | チェックポイント平均 (`average_model.py`) | 変更不要 |

#### テキスト投入方針

- 基本は**漢字混じりのまま**(モデルに読みを学習させる。これが本 FT の目的)。
- 頑健性向上のオプション: 一部(例 10–30%)を Phase 1 の `ja_frontend` でカタカナ化して混合し、
  カタカナ入力(既存の公式作法)との両対応を維持する。
- 将来: 日本語 instruct(「日本語で話してください」等)の追加学習も同じ枠組みで可能。

#### 設定上の注意 (`conf/cosyvoice3.yaml`)

- `filter: token_max_length: 200` — 日本語は Qwen BPE で概ね 1 トークン/文字。**200 文字超の発話は黙って捨てられる**ため、保有データのセグメント長分布を事前確認する(長尺中心だと学習データが激減する)。
- `max_length: 6000`(音声フレーム)/ `truncate_length: 24960` も同様にセグメント長依存。
- 学習環境は Docker GPU 環境(`docker/Dockerfile.gpu`)を想定。0.5B LLM の FT は 600h なら単一〜少数 GPU で数日オーダー。8,000h は複数 GPU + エポック設計の見直しが前提。

#### 使用データセット: `ayousanz/moe-speech-plus` (HF, gated) — 2026-07-07 確定

スタジオ収録のゲームキャラクター演技音声(プロ声優、ノイズ・BGM なし)。日本語 TTS 向けに機械フィルタ済み。

| 項目 | 内容 | CV3 学習への適合 |
|---|---|---|
| 規模 | 約 600 時間 / キャラ別 zip 約 1,004 個(各 52MB〜1.6GB、Git LFS) | 600h 先行計画とそのまま一致 |
| 音声 | 44.1kHz 16bit mono WAV、**1 発話 2〜15 秒** | ✓ 30 秒制限・`token_max_length: 200` を自然にクリア |
| 構成 | `data/{uuid}/wav/{uuid}_NNN.wav` + **同名 JSON**(zip はキャラ =uuid 単位) | utt2spk = uuid でそのまま作れる |
| 書き起こし | JSON 内に 2 系統: `anime_whisper_transcription` / `parakeet_jp_transcription`(**漢字かな混じり、句読点あり**) | ✓ 漢字直入力学習の目的に合致。ASR 疑似ラベルなので品質フィルタ推奨 |
| 品質指標 | `speechMOS` (UTMOS v2)、`duration`、感情ラベル 3 系統 | フィルタ・instruct 拡張に利用可 |
| メタ | ルートに `info.csv`(uuid, ファイル数, 総分数, f0_mean) | 話者バランス調整に利用可 |
| ライセンス | 著作権法 30 条の 4(情報解析)目的限定、**再配布禁止**、gated | 学習利用は OK。学習済みモデルの公開可否はライセンス要確認 |

**学習レシピ — 実装済み (2026-07-07): `examples/moe_speech/cosyvoice3/`**
- `local/prepare_data.py`: zip 並列展開 → JSON メタ読み込み → フィルタ
  (speechMOS ≥2.5、2 系統書き起こしの CER ≤0.2〔句読点・スタイル記号は無視して比較〕、duration 1〜29 秒)
  → text は `parakeet_jp_transcription` 既定(`--transcription anime_whisper` で切替)→ 話者ごと 1 発話を dev に分離 →
  train/dev の kaldi 形式ファイル + instruct(`You are a helpful assistant.<|endofprompt|>`)を UTF-8 で出力。
- `run.sh`: stage -1 (hf download) → 0 (prepare) → 1 (campplus) → 2 (speech token) → 3 (parquet) → 5 (**llm のみ**学習) → 6 (平均化)。
- `conf/`: libritts の cosyvoice3.yaml / ds_stage2.json を流用。
- 検証: ユニットテスト 10 件 + 合成 zip データでのエンドツーエンド確認(フィルタ 3 種・dev 分割・UTF-8 出力を確認済み)。
- `tools/make_parquet_list.py` 修正済み: UTF-8 + `maxsplit=1`(a05b026 移植)に加え、
  **`job()` 内の `spk_list` 未定義 NameError**(apply_async に握りつぶされ parquet が黙って欠落する上流バグ)を修正し、
  ワーカー例外を `result.get()` で伝播するようにした。

#### 学習インフラ: vast.ai — GPU 選定 (2026-07-07 時点の実勢価格)

API キーは `.env`(`VAST_API_KEY`、gitignore 済み)。オファー相場(verified, reliability>0.98, disk≥400GB, down≥500Mbps):

| 構成 | 実勢価格 | 特徴 |
|---|---|---|
| **1x H100 SXM(本命)** | **~$2.2/hr** | 回線 7〜8Gbps(184GB+ の DL が速い)、ディスク 1.5〜2.6TB、DDP 不要で単純。0.5B の FT には十分過ぎる性能 |
| 4x RTX 4090(対抗) | ~$1.2–1.3/hr | 前処理(speech token 抽出)を 4 並列シャードできる。DDP 学習。$/FLOP は最良クラス |
| 1x RTX 4090(節約) | ~$0.32/hr | 総額 <$10 で完走可能だが wall-clock 3〜4 倍 |
| 1x H200 | ~$3.0/hr | VRAM 141GB は 0.5B に不要。割高 |
| RTX 5090 / Blackwell 系 | - | **回避**: torch>=2.7 必須で、CV3 の音声破損 issue #1886 と requirements の torch==2.3.1 ピンに抵触 |

**推奨: 1x H100 SXM**。総コスト試算(概算): DL ~1h + 前処理(campplus + speech token 抽出、~40 万発話)4〜8h + LLM 学習(~70M トークン/epoch × 3)2〜4h ≒ **$25〜40**。
ユーザー方針「総コストが同程度なら速い GPU」に合致。同時間帯の 4x4090 ($1.23/hr) は $/FLOP でわずかに勝るが、DDP・シャーディングの手間と釣り合わない差。
価格・在庫は変動するため**インスタンス作成直前に `vastai search offers` で再検索**する(上記オファー ID は失効前提)。
CUDA ドライバは 12.5+ のホストで torch 2.3.1 (cu121) がそのまま動く。8,000h スケール時は 4x〜8x H100 または 8x4090 で前処理・学習を並列化する。

### Phase 4(任意): 品質の追求

- アクセント句情報の入力への付与(pyopenjtalk の acc 情報)、ユーザー辞書、yomikata による読み分け強化。
- `forward_dpo` の CV3 対応修正(DPO 学習をやる場合の前提)。

---

## 5. 環境メモ (Windows ローカル)

- **uv はプロジェクト管理で運用する**: 依存はすべて pyproject.toml + uv.lock(`uv sync` で環境構築)。
  依存追加は **`uv add <pkg>`**(`uv pip` は使わない方針、ユーザー指示 2026-07-07)。
- **requirements.txt は廃止 (2026-07-07)**: 全 42 依存を pyproject.toml に移行(プラットフォームマーカー、
  PyTorch cu121 / onnxruntime-cuda-12 のカスタムインデックス、deepspeed・tensorrt の静的メタデータ、
  openai-whisper の setuptools<81 ビルド依存を含む)。Docker は `uv export --frozen` 経由で pip インストール。
  vastai CLI は pillow ピン衝突のため依存に含めず **`uvx vastai`** で隔離実行する。
- `pretrained_models/` は未ダウンロード。pyopenjtalk は初回実行時に辞書 (~22MB) を自動ダウンロードする。
- `.env`(`VAST_API_KEY`)は .gitignore 追加済み。**Claude Code は .env を読み書きできない**(セキュリティ設定)ため、作成・編集はユーザーが行う。
- HF は `hf` CLI でログイン済み(gated データセットのダウンロード可)。

## 6. 学習高速化・バージョンアップ調査 (2026-07-07)

### 6.1 バージョンアップマトリクス(実施済み)

| 項目 | 旧 | 新 | 根拠・制約 |
|---|---|---|---|
| Python | 3.10 | **3.12** | pyopenjtalk 0.4.1 が cp312 ホイール無しでブロックしていたが、**pyopenjtalk-plus**(VOICEVOX 系フォーク、cp310〜cp314 全ホイール、import 名は同じ `pyopenjtalk`)へ切替して解禁。3.13 は他依存の残リスクがあり見送り |
| torch | 2.3.1+cu121 | **2.11.0+cu128** | 最新 stable は 2.12.1 だが **torchaudio の最新が 2.11.0** のため 2.11 系が実質上限。cu128 はドライバ CUDA 12.x 全体で動く(vast.ai の H100 ホストは 12.5〜13.2)。cu130 は 2.12 系のデフォルトで今回は不要 |
| transformers | 4.51.3 | **4.51.3 据え置き** | **transformers 5.x は Qwen2 実装変更(attention interface / RMSNorm / KV cache)の数値ドリフトで CV2/CV3 の自己回帰推論が雑音化**([issue #1886](https://github.com/FunAudioLLM/CosyVoice/issues/1886)、未修正のまま stale close)。5.x は禁止、4.x 内のバンプも学習後の推論検証とセットでのみ行う |
| onnxruntime(-gpu) | 1.18.0 (Azure feed) | **1.22.0 (PyPI)** | 1.19 以降 PyPI 本体が CUDA12 ビルドになり Azure フィード不要に。1.27 は CUDA13 専用 + py3.11+ でホストドライバ制約が増えるため見送り |
| pyworld | 0.3.4 | 0.3.5 | cp312/313 ホイール対応 |
| grpcio(-tools) | 1.57.0 | 1.62.3 | cp312 ホイール。1.67+ は protobuf>=5 要求で protobuf==4.25 ピンと衝突するため 1.62 系 |
| matplotlib | 3.7.5 | 3.9.4 | cp312 ホイール |
| onnx | 1.16.0 | 1.17.0 | cp312 ホイール |

- Matcha-TTS (third_party) は `torch.stft(..., return_complex=True)` を使用しており新 torch で問題なし(確認済み)。
- `torch.load` は torch 2.6 から `weights_only=True` がデフォルト。学習チェックポイントはテンソル+プリミティブのみで通る想定だが、vast.ai 上での初回実行時に要確認。
- `torch.cuda.amp.*` は deprecated 警告が出るが 2.11 でも動作する。

### 6.2 精度設定 (bf16 / fp16) — 調査結果

- **`--use_amp` + torch_ddp で既に bf16 になっている**(`train_utils.py`: `dtype = 'bf16' if args.use_amp else 'fp32'`)。run.sh は `--use_amp` 指定済み。
- H100 では bf16 が正解(fp16 と同速で、loss scale の不安定性がない)。fp16 は deepspeed エンジン + ds_config 経由でのみ選択される。
- **GradScaler は bf16 では純オーバーヘッドだったため廃止(2026-07-07 実装)**: autocast の有効化を scaler ではなく dtype で判定するよう変更し、scaler は fp16 時のみ生成。副次効果として **CV パスも bf16 で走る**ようになり約2倍高速(従来は fp32 だった)。

### 6.3 attention — 調査結果

- `Qwen2Encoder` は `Qwen2ForCausalLM.from_pretrained()` をデフォルト設定で呼んでおり、transformers 4.51 では **SDPA が既定で有効**。ただし 2D パディングマスクを渡すため SDPA の flash バックエンドは不適格で、4D float マスク経由の効率的なカーネルにディスパッチされる。
- `flash_attention_2` は診断上は正しい改善候補(2-5%)だが、torch 2.11+cu128 向けビルド済みホイールが保証されず、ソースビルド 30 分〜2 時間がスポット再起動ごとに再発しうるため**却下**(8,000h 学習で再検討)。

### 6.4 学習スループットの実装済み変更 (ultracode 監査 2026-07-07、6視点×懐疑検証で25件採用)

コード側(全レシピ共有、数値は不変):

| 変更 | 場所 | 期待効果 |
|---|---|---|
| **死んだ lm_head 行列積のスキップ** | `llm.py` Qwen2Encoder.forward が Qwen2Model バックボーンを直接呼ぶ(`use_cache=False` も指定)。151,936 語彙への [B*T,896] 射影が毎ステップ計算→破棄されていた | **5-10%/step + VRAM 2.6-5GB 解放**(hidden states はビット一致、テストで検証済み) |
| **LabelSmoothingLoss の CE 高速路** | smoothing==0 時に `F.cross_entropy` に短絡。旧実装は (B*T,6761) の dense fp32 一時テンソル数本 + forward 中の `.item()` GPU→CPU 同期 | 3-5%/step + 1-2GB(値・勾配一致をテストで検証済み) |
| **bf16 で GradScaler 廃止** | `train.py`/`train_utils.py`、autocast を dtype 基準に | 2-3%/step + CV が bf16 化 |
| **fused Adam** | `train_utils.py`、全パラメータが CUDA 上のときのみ `fused=True` | 3-8%/step(~10 回の foreach パス→1 カーネル) |
| **WORLD_SIZE==1 で DDP スキップ** | `wrap_cuda_model` + executor/save_model の 5 箇所を isinstance ガード | 2-5%/step(find_unused 走査・バケット copy・no-op allreduce 排除) |
| ws>1 の DDP 改善 | `gradient_as_bucket_view=True`、BatchNorm 無しなら `broadcast_buffers=False` | 将来のマルチ GPU 用 |
| **`--torch_compile` フラグ(オプトイン)** | 内側 Qwen2Model のみ in-place `compile(dynamic=True)`(外側 forward は Python ループ/.tolist() だらけで不適)。state_dict キー汚染なし | 8-15%/step 見込み、4090 で要実測。`optimize_ddp=False`, `cache_size_limit=16` |
| TensorBoard 書き込み間引き | log_per_step を log_interval に同期 | ~1% |
| `--onnx_path` ガード | 未指定時に env を立てない。moe_speech の stage 5 から削除(トークン事前計算済みなのに ORT CUDA セッションが VRAM 0.5-1.5GB 占有していた) | VRAM 解放 + 空トークンフィルタ復活 |

moe_speech レシピ側:

| 変更 | 内容 | 期待効果 |
|---|---|---|
| **波形フリー llm データパイプライン** | `parquet_opener` に columns プルーニング(audio_data 列 ≒ シャードの 99% を読まない)+ `filter_speech_token`/`sort_by_speech_token`/`dynamic_batch_llm`/`padding_llm`(すべて speech_token 長 = mel/2 基準)。wav デコード・リサンプル・mel/whisper fbank を全廃 | **ワーカー CPU ~30-60ms/サンプル → <1ms、エポックあたり parquet 読取 ~200GB → ~1-2GB**。ローダー律速なら 25-50% |
| `max_frames_in_batch: 15000 → 30000` | H100 80GB 向け(単位は 50Hz mel フレーム、30000=600 秒/バッチ)。**24GB カード(4090)では 15000 に戻すこと** | ステップ数半減、15-30%/epoch |
| `max_epoch 10 → 5` | 600h の FT は 3-5 エポックで CV loss が平坦化する想定。まだ下がっていれば `--checkpoint` 再開で延長 | 最大 50% |
| `save_per_step 2000 → 1000` | バッチ倍増に合わせ壁時計での保存間隔を維持 | スポット耐性 |
| `prefetch 100 → 8` | prefetch はワーカー毎: 100×8=800 バッチ(~10GB+ の pinned RAM)は無意味でページング事故のもと | 事故防止 |
| dev parquet シャード数修正 | dev を `ceil(N/num_workers)` 発話/シャードで 8 シャード化。1 シャードだと DistributedSampler の複製で**全ワーカーが dev 全体を評価し CV が 8 倍**になっていた | CV 1/8 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 動的バッチの形状ばらつきによる断片化対策 | 安定性 |

前処理の高速化 (2026-07-07 追加、学習よりも前処理が費用の過半を占めるため):

| 変更 | 内容 | 期待効果 |
|---|---|---|
| **stage 2 セッションプール** | `tools/extract_speech_token.py --num_sessions N`: N 個の ONNX セッション(各自の CUDA ストリーム)をキューで貸し出し、**バッチ 1 のまま発話間並列**。単一セッション×16 スレッドでは GPU 上で直列化していた。パディング無しなので**数値は参照実装と同一** | stage 2 の 2-5h → 推定 0.7-1.5h (2-4x) |
| Resample キャッシュ | stage 1/2 で `torchaudio.transforms.Resample` を発話ごとに構築していた(sinc カーネルを 30 万回再計算)→ orig_freq 別にキャッシュ | CPU 側短縮 |
| stage 0 メタスキャン並列化 | `prepare_data.py` の 30 万件 JSON 読み+フィルタ直列ループを ProcessPool 化 (`scan_chunk`) | 10-20 分 → 2-3 分 |
| **stage 3 `--exclude_audio_data`** | `make_parquet_list.py` が wav 生バイト(~200GB、llm 学習は読まない)を parquet に書いていた → フラグでスキップ。**flow/hifigan 学習時はフラグ無しで parquet 再生成が必要** | ~40 分 → ~2 分 + ディスク 200GB 節約 |
| stage 1 CUDA オプション | `extract_embedding.py --provider cuda --num_sessions N`(既定は cpu のまま)、スレッド数 16→32 | コア数少ないホスト向け |
| stage 2 バッチ ONNX 抽出 (opt-in) | `local/extract_speech_token_batch.py`。**バッチ内ゼロパディングでトークンがずれるため既定から外した**(§6.5)。実データで一致検証が取れれば再昇格 | (保留) |

前処理見積もりの更新: stage 2 が 0.7-1.5h に縮み、stage 0/3 が数分になるため、**H100 での前処理合計は ~4-8h → ~2-3.5h、総費用見込みは $20-35 → $13-22** に低下。

**却下(検証で落ちたもの)**: flash-attn 2(§6.3)、find_unused_parameters=False 単独(DDP スキップに包含)、monitored_barrier/NCCL チューニング(ws=1 では <0.1%)、`.to(device, non_blocking=True)`(直後の .cpu()/.item() 同期で無意味)。

### 6.5 Linux E2E スモークテスト結果 (2026-07-07, vast.ai RTX 4090 $0.336/hr)

合成 2000 発話(2〜12 秒)で clone → uv sync → モデル DL → stage 0-3 → stage 5(2 エポック ×2 回)を実施。**本番前に 4 つの地雷を検出・修正**:

| 問題 | 症状 | 修正 |
|---|---|---|
| **torchaudio 2.9+ の torchcodec 依存** | `torchaudio.load/save` が ImportError → 前処理が例外を握りつぶし**全発話が空トークン**の「見かけ成功」 | `audio_load/audio_save`(soundfile 直叩き)を file_utils に追加し学習経路の全呼び出しを置換 (`812609f`) |
| **torch 2.11 で `ProcessGroup.options` 削除** | 学習ループ 2 バッチ目で AttributeError | cosyvoice_join が `--timeout` 引数から timedelta を渡す (`3e020f5`) |
| run.sh がステージ失敗を握りつぶす | 半壊した前処理成果物で後段が走る | 全ステージに `\|\| exit 1` |
| バッチ ONNX トークン抽出の数値ずれ | バッチ内ゼロパディングで**約 1/4 の発話のトークンが 8〜25% 相違**(先頭付近から)。約半数は完全一致(長さ数え方の差のみ) | トークンは学習ターゲットのため **stage 2 の既定をバッチ 1 ツールに戻す**。バッチ版は実データ検証まで opt-in |

**torch.compile 判定(採用確定)**: eager vs `--torch_compile`、各 2 エポック 740 バッチ、max_frames 8000:

| 指標 | eager | compile |
|---|---|---|
| エポック 1(ウォーム)実時間 370 バッチ | 63.8s | **53.0s (-17%)** |
| 定常ステップ中央値 | 165.5ms | **137.0ms (-17.2%)** |
| recompile イベント(756 ステップ) | — | **2 回のみ**(shape bucket 1 + eval graph 1、定常での再発なし) |
| compile ウォームアップ | — | ~90 秒(一回きり、時間単位の本番では無視できる) |
| 初回 loss/acc | 2.0350 / 0.3940 | 2.0353 / 0.3923(bf16 ノイズ内で一致) |

その他の確認事項: loss 2.04→1.17 / acc 0.39→0.60 と学習が正常進行、checkpoint は `module.`/`_orig_mod` 汚染なしで事前学習 llm.pt とキー完全一致(epoch/step メタデータ有り、再開可)、CV は bf16 で dev 4 発話 ~1 秒。**VRAM 実測: max_frames 15000 は 24GB で OOM**(22.4GB 割当)→ 24GB カードのガイダンスは 8000。H100 の 30000 は本番前に ~200 ステップのプローブで確認すること。

## 7. 参考リンク

- [issue #1705: CV3 学習の語彙サイズ不一致](https://github.com/FunAudioLLM/CosyVoice/issues/1705)
- [issue #303: 日本語誤読](https://github.com/FunAudioLLM/CosyVoice/issues/303)
- [issue #152: 日本語長文が分割されない](https://github.com/FunAudioLLM/CosyVoice/issues/152)
- [Fun-CosyVoice3-0.5B-2512 モデルカード](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk) / [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) / [yomikata](https://github.com/passaglia/yomikata)
- ブランチ内既存ドキュメント: `origin/feature/japanese-hybrid-preprocessing:docs/japanese_improvement_guide.md`, `docs/tokenizer_comparison.md`

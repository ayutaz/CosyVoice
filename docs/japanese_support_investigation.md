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

## 6. 参考リンク

- [issue #1705: CV3 学習の語彙サイズ不一致](https://github.com/FunAudioLLM/CosyVoice/issues/1705)
- [issue #303: 日本語誤読](https://github.com/FunAudioLLM/CosyVoice/issues/303)
- [issue #152: 日本語長文が分割されない](https://github.com/FunAudioLLM/CosyVoice/issues/152)
- [Fun-CosyVoice3-0.5B-2512 モデルカード](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk) / [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) / [yomikata](https://github.com/passaglia/yomikata)
- ブランチ内既存ドキュメント: `origin/feature/japanese-hybrid-preprocessing:docs/japanese_improvement_guide.md`, `docs/tokenizer_comparison.md`

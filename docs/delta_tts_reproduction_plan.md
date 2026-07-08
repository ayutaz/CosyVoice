# DELTA-TTS 再現実験計画

- 論文: **DELTA-TTS: Adapting Autoregressive Model into Diffusion Language Model for Text-to-Speech**
- arXiv: [2607.04140](https://arxiv.org/abs/2607.04140) (v1, 2026-07)
- 著者: Junwon Moon, Seungbeom Kim, Yejin Lee, Hoseong Ahn, Sewoong Park, Heeseung Kim, Kyuhong Shim ほか
- 公式実装: **なし**(論文にコード・デモページの記載なし。フルスクラッチ再現が必要)
- 作業ブランチ: `feature/delta-tts-reproduction` (`feature/japanese-katakana-frontend` から分岐)

---

## 1. 論文の概要

自己回帰型 (AR) TTS の LLM 部分を、**軽量な LoRA ベースの適応で離散拡散言語モデル (masked diffusion LM) に変換**する手法。
バックボーンは **CosyVoice3**(Qwen2-0.5B が 25Hz セマンティック音声トークンを生成 → flow matching デコーダで波形化)であり、本リポジトリの構成とそのまま対応する。

ポイントは「ARモデルをゼロから NAR に置き換える」のではなく、**事前学習済み AR バックボーンを凍結したまま、LoRA (約35Mパラメータ) と畳み込みモジュールの追加だけで拡散型の並列デコーディングを獲得する**こと。

### 主要な結果(論文の主張)

| 指標 | DELTA-TTS | CosyVoice3 (AR) |
|---|---|---|
| WER (Seed-TTS test-en) | **1.75%** | 2.02% |
| SIM | 0.688 | — |
| RTF | **0.144** | 0.475 |
| 速度向上 | 3.3× (5-10秒音声では 4.46×) | 1× |

AR より速いだけでなく WER も改善(左→右の誤り伝播・ハルシネーションが減るため)。

---

## 2. 手法の詳細

### 2.1 アーキテクチャ変更(CosyVoice3 LLM に対して)

1. **LoRA 適用**: 各 Transformer ブロックの q/k/v/o および gate/up/down projection に適用。α=128、合計 +35M パラメータ。**バックボーンは凍結**。
   - ※ rank (r) は論文に明記なし(要検討事項 → §6)
2. **因果マスク → 双方向自己注意**に変更
3. **マスクトークン [M]** を追加。埋め込みは全音声トークン埋め込みの平均で初期化
4. **Shift operation**: AR の挙動を保持する。入力側はシフトせず([M] はマスク位置 j 自体に置く)、出力側のみ AR の契約を維持 — hidden[i] が「トークン i+1」の logits を出し、マスク位置 j は hidden[j−1] から読む(Phase 0 の S2 スパイクで一次情報4件により確定。`docs/delta_tts_phase0_verification.md` B-2 参照)
5. **畳み込みモジュール**: Conformer 様式で各ブロック後に残差接続。depthwise conv + GLU + Swish、kernel size 31、dropout 0.1。局所的な音響構造の捕捉が目的(アブレーションで最大の寄与: WER 2.59% → 1.61%)

### 2.2 訓練(masked diffusion)

- 入力レイアウト: `[SOS, t_inst, t_prompt, t_target, TASK, s_prompt, s_target, EOS]`
- **マスクは s_target(ターゲット音声トークン)のみ**に適用。テキストとプロンプト音声トークンは常に可視
- 訓練時: マスク確率 t ∈ (0,1] をサンプリングし、s_target の各トークンを確率 t で [M] に置換
- **1/t 加重訓練目標**: マスクされた位置の CE 損失を 1/t で加重(ELBO から導出)。低マスク率(=デコーディング後半に相当)のステップに大きな重みが付く

### 2.3 推論(信頼度順序デコーディング)

1. ターゲット長分の [M] 列を用意し、T ステップ(デフォルト **T=16**)で反復的にアンマスク
2. 各ステップで全マスク位置を並列予測し、**top-p (=0.8) nucleus sampling** で候補を取得
3. サンプルされたトークンの確率を信頼度とし、**信頼度上位の候補のみ確定**、残りは再マスク
4. 各ステップの確定数は**時間シフトスケジュール**で決定:
   `c_n = μ·(n/T) / (1 + (μ−1)·(n/T))`、**μ=0.3**
   → 序盤は少数のみ確定し、終盤に確定を集中させる(coarse-to-fine)
5. CFG は不使用

### 2.4 ターゲット長の決定(NAR ゆえ事前に必要)

ルールベース: `目標音声トークン数 = r_prompt × W_target`、ただし `r_prompt = N_prompt_audio / W_prompt`(プロンプトの「音声トークン数/文字数」比をターゲットテキストに外挿)。
**論文の主評価(Table 1: WER 1.75% / SIM 0.688)はこのルールベース長で行われている**(GT 長は ablation 内の比較変種で 1.63% / 0.686。Phase 0 検証で確定)。EOS による動的停止はなく、固定長生成。

---

## 3. 論文の実験設定

| 項目 | 設定 |
|---|---|
| 訓練データ | **LibriTTS 585h**(英語) |
| GPU | A100 ×1、bfloat16 |
| Optimizer | AdamW、lr=1e-4(定数)、linear warmup 2,000 steps |
| バッチ | 実効 16(勾配蓄積) |
| 総ステップ数 | 明記なし(要探索) |
| 評価セット | Seed-TTS test-en (1,088件)、LibriSpeech-PC test-clean Subset B (1,127件) |
| 評価指標 | WER (Whisper-large-v3)、SIM (WavLM-large ECAPA-TDNN)、UTMOS、RTF、CMOS/SMOS |

### アブレーション(表5、Seed-TTS test-en WER)

| 構成 | WER |
|---|---|
| Naive 変換(LoRA のみ) | 3.01% |
| + 時間シフトスケジュール | 2.59% |
| + 畳み込みモジュール | **1.61%** |
| + ルールベース長(最終形) | 1.75% |
| (参考) LoRA でなく全パラメータ FT | 1.97%(過学習傾向) |
| (参考) 知識蒸留アプローチ | 2.37% |

---

## 4. 再現実験で「何をしたいのか」

### 目的

1. **忠実再現**: CosyVoice3 + LibriTTS で論文の変換手法を実装し、WER / SIM / RTF が論文値(WER 1.75%、RTF 0.144、3.3×高速化)に近づくか検証する
2. **日本語への展開**(本プロジェクト固有の動機): 現ブランチで構築した日本語対応 CosyVoice3(カタカナフロントエンド + moe-speech-plus 600h FT 済みモデル)に同じ変換を適用し、**日本語ゼロショット TTS の高速化(3〜4×)と CER 維持・改善**が成立するか確認する
3. 副次的に、既存の SSD (speculative decoding) ブランチとの**速度・品質トレードオフ比較**の土台を作る

### スコープ(Phase 分割)

**Phase 1: 実装(GPU 不要、ローカルで進められる)**
- [ ] `DiffusionCosyVoice3LM`(仮称)の実装: `cosyvoice/llm/llm.py` の `CosyVoice3LM`(llm.py:669)を継承 or 並置
  - 双方向 attention 化(Qwen2 の causal mask 差し替え)
  - LoRA(peft 利用を想定 → `uv add peft`)
  - [M] トークン追加(speech_embedding の平均で初期化)
  - Conformer 風 conv モジュール(kernel 31, GLU, Swish, dropout 0.1, 残差)
- [ ] 訓練スクリプト: `cosyvoice/bin/train_draft.py` の構成を参考に masked-diffusion 用データローダ/損失(1/t 加重 CE)を実装
- [ ] 推論: 信頼度順序デコーディング + 時間シフトスケジュール + ルールベース長決定
- [ ] ユニットテスト(`tests/`、`.venv\Scripts\python.exe -m pytest`)

**Phase 2: 英語での忠実再現(vast.ai H100)**
- [ ] LibriTTS 585h の準備(既存の `scripts/prepare_libritts.py` が流用候補)
- [ ] CosyVoice3 公式 checkpoint 凍結 + LoRA 訓練(A100×1 相当 → H100×1 で可)
- [ ] Seed-TTS test-en で WER/SIM/RTF 評価 → 論文表1・表5と突き合わせ

**Phase 3: 日本語展開**
- [ ] 日本語 FT 済みモデル(バックアップ済み checkpoint)をバックボーンとして同手法を適用
- [ ] moe-speech-plus を訓練データに、既存の日本語 CER 評価パイプライン(`docs/japanese_finetuning_report.md` 参照)で AR 版と比較

### 本リポジトリでの主な変更対象

| ファイル | 変更内容 |
|---|---|
| `cosyvoice/llm/llm.py` | 拡散版 LM クラス追加(または新規 `cosyvoice/llm/diffusion_llm.py`) |
| `cosyvoice/cli/model.py` | `CosyVoice3Model` に拡散推論パスを追加 |
| `cosyvoice/cli/cosyvoice.py` | API から拡散モードを選択可能に |
| `examples/libritts/cosyvoice3/conf/` | `cosyvoice3_delta.yaml`(仮)追加 |
| `cosyvoice/bin/` | `train_delta.py`(masked diffusion 訓練) |
| `scripts/` | 評価スクリプト(WER/SIM/RTF、`eval_ssd.py` が参考になる) |

---

## 5. 評価計画

- **英語**: Seed-TTS test-en(公開されている 1,088件)で WER(Whisper-large-v3)/ SIM(WavLM-large ECAPA-TDNN)/ UTMOS / RTF。論文の表1を再現目標とする
- **日本語**: 既存 CER 評価パス(frontend/finetune 比較で使用済み)を流用し、AR 版日本語モデルと DELTA 版で CER / SIM / RTF を比較
- **速度**: 音声長ビン(0-3s / 3-5s / 5-10s)別の speedup を計測(論文表3と同形式)
- アブレーション再現(最低限): naive変換 → +時間シフト → +conv の3点で WER 変化を確認

## 6. 未確定事項・リスク

※ 事前検証の詳細と解決済み項目は `docs/delta_tts_phase0_verification.md` を参照。

| 項目 | 状態 | 対応方針 |
|---|---|---|
| LoRA rank | **解決(S3)**: r=64 で trainable=35,192,832 を実測、逆算値と厳密一致 | peft 0.19.1 追加済み。仕様は phase0 doc B-3 |
| 双方向 attention 化 | **解決(S1)**: 4D additive マスクで双方向化を数値検証済み。flash_attention_2 は不可 → sdpa を使用 | 仕様は phase0 doc B-1。transformers 5.x へのピン解除は厳禁 |
| Shift operation の厳密な定義 | **解決(S2)**: 入力は無シフト、[M] は位置 j、logits は hidden[j−1] から読む(出力側の右シフト) | 訓練/推論の擬似コードは phase0 doc B-2 |
| 訓練時の prompt/target 分割 | **解決(S4)**: 同一発話接頭辞 + テキスト非分割(t_prompt 空)方式に決定。アライメント不要 | `--prompt_mode` フラグで代替方式も切替可能に実装 |
| CosyVoice3 checkpoint | **解決**: `pretrained_models/Fun-CosyVoice3-0.5B` が手元にあり、`llm.pt`(非RL)を使用 | — |
| 訓練スクリプト等の雛形 | **解決**: `origin/feature/speech-speculative-decoding` に prepare_libritts.py / train_draft.py / eval_ssd.py あり | `git checkout origin/... -- <path>` で取り込み |
| 推論の長さ決定 | **解決(S2検証)**: 論文の主評価はルールベース長(GT長は ablation 変種) | ルールベース長を標準採用 |
| 総訓練ステップ数 | 未解決(明記なし) | 損失と検証 WER を見ながら決定。LibriTTS 585h / batch16 でエポック数を仮置き |
| Seed-TTS test-en の入手 | 手順確認済み | [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) 公式プロトコルに従う(WER: Whisper-large-v3、SIM: WavLM-large SV) |
| RTF の比較条件 | 論文は A100、手元は H100 | 自前 AR ベースライン(S5)との speedup 比で比較 |

残る細部(t の分布、lora_dropout、接頭辞率上限、conv 挿入位置の詳細、H100/bf16 での sdpa 検証)は
`docs/delta_tts_phase0_verification.md` §C を参照。

## 7. 次のアクション

Phase 0(技術スパイク S1〜S4)は完了(2026-07-08、結果は `docs/delta_tts_phase0_verification.md`)。
`peft==0.19.1` 追加済み。

1. Phase 1 実装に着手: 拡散版 LM クラス(phase0 doc B-1〜B-4 の確定仕様に従う)、
   `train_delta.py`、信頼度順序デコーディング
2. 実装が通ったらダミーデータで訓練ループのスモークテスト
3. vast.ai H100 で S5〜S7(ARベースライン RTF / LibriTTS 疎通 / 長尺一括合成)→ Phase 2 へ

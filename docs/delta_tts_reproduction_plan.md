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

### 目的(2026-07-09 再スコープ: 日本語適用のみ。英語再現は対象外)

1. **日本語モデルへの適用**: 日本語 FT 済み CosyVoice3(`checkpoints/cosyvoice3_ja/llm.pt`、
   moe-speech-plus 470h で漢字直接入力 CER 0.085 を達成済み。`docs/japanese_finetuning_report.md`)を
   凍結バックボーンとして delta 変換し、**日本語ゼロショット TTS の高速化(3〜4× 目標)と
   CER 維持(ft_kanji 0.085 → 0.10 以下を目安)**が成立するか検証する
2. 副次的に、既存の SSD (speculative decoding) ブランチとの速度・品質トレードオフ比較の土台を作る

※ 英語(LibriTTS)での論文忠実再現はユーザー判断で**スコープ外**とした(2026-07-09)。
論文値(WER 1.75% 等)は §1〜3 の参照情報としてのみ保持する。英語用に作成した
`examples/libritts/cosyvoice3/conf/cosyvoice3_delta.yaml` は削除済み(必要になれば
git 履歴 `40f9e32` から復元可能)。

### スコープ(Phase 分割)

**Phase 1: 実装 — 完了(2026-07-09。詳細は `docs/delta_tts_phase1_implementation.md`)**
- [x] `DiffusionCosyVoice3LM` の実装: `cosyvoice/llm/diffusion_llm.py`(CosyVoice3LM を継承)
  - 双方向 attention 化(4D additive マスク)/ LoRA r=64(peft)/ [M] は独立 nn.Parameter /
    ConformerConvModule(k=31, GLU, SiLU, dropout 0.1, 残差, 最終pwゼロ初期化)
- [x] 訓練スクリプト: `cosyvoice/bin/train_delta.py` + `cosyvoice3_delta.yaml`(1/t 加重 CE、
  ConstantWithWarmupLR で論文の lr=1e-4 定数を再現)
- [x] 推論: `inference_diffusion`(信頼度順序デコーディング + 時間シフトスケジュール + ルールベース長)
- [x] ユニットテスト 46 件 + 0.5B 実機スモーク(`scripts/spikes/s8_delta_smoke.py`)。
  trainable が論文と一致(LoRA 35,192,832 / conv 58,641,408)することを実機確認済み
- 未実施(Phase 2 冒頭へ): `cli/model.py` / `cli/cosyvoice.py` への配線、長さルールの文字数基準化

**Phase 2: 日本語 delta 訓練と評価(vast.ai H100)**
- [x] 日本語用 delta 設定: `examples/moe_speech/cosyvoice3/conf/cosyvoice3_delta.yaml`
  (moe_speech の波形フリー llm パイプライン + instruct 列 + 論文の訓練レシピ)
- [x] instruct 領域対応: 日本語 FT は全発話 `You are a helpful assistant.<|endofprompt|>` 付きで
  学習されているため、delta forward が instruct を text 領域の先頭に連結(可視・損失対象外)
- [x] `cli/model.py` への `inference_diffusion` 配線(llm_job の duck-typing dispatch)と
  変換ヘルパー `DiffusionCosyVoice3LM.from_ar()`(2026-07-09)
- [x] `scripts/eval_ja_cer.py` に delta_katakana / delta_kanji 経路 + `--delta_checkpoint` +
  全経路の平均 RTF 記録を追加(2026-07-09)
- [x] e2e スモーク: ローカル RTX 4070 Ti SUPER + 日本語FTバックボーンで
  frontend→拡散デコード→flow→hift の全経路 PASS(`scripts/spikes/s9_delta_e2e_smoke.py`)
- [x] moe-speech-plus parquet の用意(vast.ai 上で再生成。**波形フリー parquet 523MB を
  ローカルにアーカイブ済み** → 次回は前処理丸ごとスキップ可)
- [x] delta 訓練(2026-07-09、H100、~$13): ピークは step 20k-30k、以降過学習 → 54k で停止
- [x] 評価完了: **delta(平均化 20k-30k)CER 0.106 / RTF 0.945 = AR 比 2.0×速**
  (ft_kanji 0.0854/RTF 1.89、base_katakana 0.0968)。目標(≤0.10、3×)にわずかに未達。
  **結果詳細と改善候補: `docs/delta_tts_phase2_results.md`**

(英語での忠実再現フェーズは廃止 — 上記「目的」の注記を参照)

### 本リポジトリでの主な変更対象

| ファイル | 変更内容 |
|---|---|
| `cosyvoice/llm/llm.py` | 拡散版 LM クラス追加(または新規 `cosyvoice/llm/diffusion_llm.py`) |
| `cosyvoice/cli/model.py` | `CosyVoice3Model` に拡散推論パスを追加 |
| `cosyvoice/cli/cosyvoice.py` | API から拡散モードを選択可能に |
| `examples/moe_speech/cosyvoice3/conf/` | `cosyvoice3_delta.yaml` 追加(日本語 delta 訓練用)|
| `cosyvoice/bin/` | `train_delta.py`(masked diffusion 訓練) |
| `scripts/` | 評価スクリプト(WER/SIM/RTF、`eval_ssd.py` が参考になる) |

---

## 5. 評価計画

- **日本語(主)**: `scripts/eval_ja_cer.py` の評価パス(漢字混じり20文 × whisper large-v3 CER)に
  delta 経路を追加し、AR ベースライン(ft_kanji **0.085** / base_katakana 0.097)と比較。
  合格目安: delta_kanji CER ≤ 0.10 かつ RTF 3× 以上の高速化
- **速度**: 音声長ビン(0-3s / 3-5s / 5-10s)別の speedup を計測(論文表3と同形式)。
  ベースラインは同一ハードウェアでの日本語 FT AR モデル
- アブレーション(余力があれば): naive変換 → +時間シフト → +conv の3点で CER 変化を確認

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
| 総訓練ステップ数 | 未解決(論文に明記なし) | CV loss を見ながら決定。日本語 AR FT は同じ 470h で 3-5 エポックで平坦化した実績を目安に |
| RTF の比較条件 | 論文は A100、手元は H100 | 自前 AR ベースライン(S5)との speedup 比で比較 |

残る細部(t の分布、lora_dropout、接頭辞率上限、conv 挿入位置の詳細、H100/bf16 での sdpa 検証)は
`docs/delta_tts_phase0_verification.md` §C を参照。

## 7. 次のアクション

Phase 0(技術スパイク、2026-07-08)と Phase 1(実装、2026-07-09)は完了。
結果は `docs/delta_tts_phase0_verification.md` / `docs/delta_tts_phase1_implementation.md`。
2026-07-09 に**日本語適用を主目的に再スコープ**(instruct 対応・日本語用 config 追加済み)。

**Phase 2 完了(2026-07-09)**: delta(平均化)= CER 0.106 / 2.0×速。
結果と運用記録は `docs/delta_tts_phase2_results.md`。

次の改善イテレーション候補(同 doc §5):
1. 低 lr(1e-5)で step 25k から短い追い込み再訓練(parquet アーカイブ利用で ~$3)
2. 平均化の窓の最適化(SWA 的な近傍平均)
3. 推論パラメータ探索(T=8、mu、top_p、length_scale)— GPU 不要

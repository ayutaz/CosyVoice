# DELTA-TTS 再現: Phase 0(事前技術検証・調査)

Phase 1(実装)着手前に潰しておくべき技術リスクの調査結果と、残る検証項目のリスト。
計画本体は `docs/delta_tts_reproduction_plan.md` を参照。

---

## A. 調査済み・確認済みの事項

### A-1. CosyVoice3 checkpoint は手元に揃っている ✅

`pretrained_models/Fun-CosyVoice3-0.5B/` に以下を確認:

- `llm.pt` / `llm.rl.pt`(LLM 2種。論文は RL に言及なし → **`llm.pt`(非RL)をベースにする**)
- `flow.pt`, `hift.pt`(拡散変換では凍結・共用)
- `speech_tokenizer_v3.onnx` / `speech_tokenizer_v3.batch.onnx`(訓練データのトークン抽出に必要)
- `CosyVoice-BlankEN/`(Qwen2 バックボーン、HF形式)

設定値: `speech_token_size=6561`、語彙 6561+200=6761、hidden 896、24層、intermediate 4864、KVヘッド2(GQA、KV次元128)。

### A-2. 双方向 attention は transformers 4.51.3 で実現可能 ✅

`Qwen2Model` は **4D の attention_mask を渡すとそのまま使用する**
(`.venv/.../modeling_qwen2.py:698` — `if attention_mask.dim() == 4: do nothing`)。
全可視の 4D マスクを自前で構築して渡せば、モデル改造なしで因果マスクを無効化できる。

- 注意: このリポジトリは **transformers==4.51.3 に意図的にピン留め**(5.x は Qwen2 数値が変わり CosyVoice 推論が壊れる、pyproject.toml の NOTE 参照)。拡散化もこのピンの範囲内で行うこと
- 残る確認は S1 スパイク(数値検証)のみ

### A-3. LoRA rank は r=64 でほぼ確定(論文未記載だが逆算一致)✅

論文記載: α=128、LoRA +35M。Qwen2-0.5B の7 projection(q/k/v/o/gate/up/down)で計算すると

- 1層あたり: r×(1792+1024+1024+1792+5760+5760+5760) = **22,912r**
- 24層合計: 549,888r → **r=64 で 35.19M ≒ 論文の 35M** ✔

さらに論文本文に **「trainable 94M = LoRA 35M + conv 59M」** の内訳記載を発見。
conv 59M ÷ 24層 ≈ 2.46M/層 = Conformer conv(pointwise 896→1792 GLU + depthwise k=31 + pointwise 896→896 + norm)と一致 → **conv モジュールは全24層に挿入**で整合。

### A-4. SSD ブランチの資産はリモートから取り込み可能 ✅

ローカルブランチには無いが `origin/feature/speech-speculative-decoding` に存在:

| ファイル | 流用先 |
|---|---|
| `scripts/prepare_libritts.py` | LibriTTS データ準備(Phase 2) |
| `cosyvoice/bin/train_draft.py` | `train_delta.py` の雛形(訓練ループ・DDP・Windows対応済み) |
| `scripts/eval_ssd.py` | RTF/速度評価スクリプトの雛形 |
| `examples/libritts/cosyvoice3/conf/cosyvoice3_draft.yaml` | `cosyvoice3_delta.yaml` の雛形 |

取り込みは `git checkout origin/feature/speech-speculative-decoding -- <path>`。

### A-5. 推論の統合点は既存の generator インターフェースに載る ✅

`CosyVoice3Model.llm_job`(`cosyvoice/cli/model.py:101`)は `self.llm.inference(...)` の
**トークン generator を消費する**構造。拡散デコーダは「T ステップ反復の完了後に全トークンを
一括 yield する generator」として実装すれば、`stream=False` 経路(全トークン→token2wav 一括)に
無改造で載る。flow/HiFT は共用なので RTF 比較も公平。

### A-6. 評価リソースはすべて公開されている ✅

- [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval): test-en 1,088件 + WER(Whisper-large-v3)+ SIM(WavLM-large finetuned SV モデル)の公式プロトコル。論文と同一
- UTMOS: SpeechMOS 等の公開実装で再現可
- LibriSpeech-PC test-clean Subset B(第2ベンチマーク)も公開リストあり

### A-7. 参考にできるオープン実装(DELTA-TTS 公式実装は無し)✅

| リポジトリ | 参考にする点 |
|---|---|
| [HKUNLP/DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA)(ICLR2025) | **AR→拡散適応の本家**。shift operation・attention 変換の実装 |
| [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) | 1/t 加重 masked diffusion 損失、confidence remasking デコーディング |
| [Dream 7B](https://hkunlp.github.io/blog/2025/dream/) | AR初期化からの拡散訓練の知見(学習率設定が重要) |
| LLaDA-TTS (arXiv 2603.26364) | masked diffusion TTS の評価設定の先行例 |

---

## B. Phase 1 前に推奨する技術スパイク(優先度順)

### S1: 双方向 attention の数値検証(~半日、CPU/ローカル可)【ブロッカー】

4D 全可視マスクを `Qwen2Encoder` に渡し、以下を確認する小スクリプトを書く:

- [ ] 末尾トークンを変えると**先頭位置の hidden state が変わる**(=双方向になっている)
- [ ] sdpa / eager 両実装で出力が一致する(数値許容誤差内)
- [ ] `use_cache=False` で KV キャッシュ経路を踏まないこと

### S2: shift operation の定義確定(~半日)【ブロッカー】

2回の論文要約で「位置 i が i+1 を予測」「i が i を予測」と食い違いが出た。実装前に確定が必要:

- [ ] DiffuLLaMA のコードで shift の実装(AR の next-token 重みを再利用するため、
      マスク位置 i の logits を位置 i−1 の hidden から読む方式のはず)を確認
- [ ] DELTA-TTS の PDF 付録を精読(ローカルに poppler を入れて PDF を読むか、
      arXiv HTML 版の該当節を精読)。時間シフト式の μ と本文の t_shift=0.3 の対応もここで確定

### S3: peft + LoRA 適用スパイク(~半日、CPU/ローカル可)【ブロッカー】

- [ ] `uv add peft`(transformers 4.51.3 と互換のあるバージョンを確認して選定)
- [ ] `Qwen2Encoder.model` に r=64 / α=128 / 対象7 projection で LoRA を適用し、
      **trainable パラメータ数 ≈ 35.19M** を確認(→ r=64 の裏取り完了)
- [ ] `Qwen2Encoder.forward` は `self.model.model(...)` とバックボーン直呼びしている
      (llm.py:238)ため、peft ラップ後もこの経路で LoRA 層が効くことを確認
- [ ] LoRA 込み state_dict の保存・ロード往復を確認

### S4: 訓練時の prompt/target 分割方式の決定(調査+設計判断)【ブロッカー】

論文は入力レイアウト `[SOS, t_inst, t_prompt, t_target, TASK, s_prompt, s_target, EOS]` と
「マスクは s_target のみ」を示すが、**訓練時にプロンプトをどう作るかは未記載**。

- 仮説(a): 同一発話のランダム接頭辞(例: 0〜50%)を prompt に割り当てる(MaskGCT / VALL-E NAR 系の標準)
- 仮説(b): プロンプトなしで全体を s_target とし、ゼロショット形式は推論時のみ
- [ ] まず (a) をデフォルトとして実装し、小規模訓練で (b) と比較できるようフラグ化しておく

### S5: AR ベースラインの RTF 計測(数時間、vast.ai GPU)【Phase 2 前で可】

- [ ] H100 で `Fun-CosyVoice3-0.5B` AR 推論の RTF を計測(end-to-end と LM 単体の両方、
      音声長ビン 0-3s / 3-5s / 5-10s 別)
- 論文の RTF(AR 0.475 / DELTA 0.144)は **A100 での値**。H100 では絶対値がズレるため、
  速度比較は自前ベースラインとの **speedup 比**で行う

### S6: LibriTTS データパイプラインの疎通(~1日、GPU)【Phase 2 前で可】

- [ ] `prepare_libritts.py` を SSD ブランチから取り込み、CV3 用(speech_tokenizer_v3)への
      改修点を洗い出す(SSD 時は CV2/tokenizer v2 主体だった可能性)
- [ ] 数発話で抽出→`cosyvoice/bin/train.py` の期待するデータ形式(parquet list)まで通す
- [ ] 585h 全体のトークン抽出の GPU 時間・ディスク量を見積もる

### S7: 長尺一括合成の確認(低コスト、S5 と同時)

- [ ] `stream=False` で 10 秒級の全トークン一括 token2wav の品質・VRAM を確認
      (拡散生成では全トークンが一度に出るため、この経路が本番になる)

---

## C. Phase 1 に持ち込む設計判断(スパイク結果待ち)

1. **[M] トークンの ID**: 語彙に +200 の特殊トークン領域(6561〜6760)があり、
   使用中は sos/eos/task/fill の4つのみ。**未使用スロット(例: 6565)を [M] に転用**すれば
   embedding テーブルの拡張が不要。埋め込みは音声トークン 0〜6560 の平均で上書き初期化(S3 で確定)
2. **推論時の logit マスク**: 拡散生成では eos/fill/sos/task/[M] を予測させないよう
   特殊トークンの logits を −inf にする処理が必要
3. **日本語展開時(Phase 3)の長さルール**: 論文のルールベース長は「単語数」基準で日本語に
   そのまま使えない。モーラ数または文字数ベースの比率に置き換える
4. **訓練対象パラメータ**: LoRA(35M)+ conv(59M)+ [M] 埋め込み。バックボーン・
   text embedding・llm_decoder は凍結(論文準拠)

## D. 判定

**S1〜S4 が Phase 1 の実装ブロッカー**(すべてローカル/CPU で完結、合計 2〜3 日想定)。
S5〜S7 は GPU が必要だが Phase 2 開始前までに済めばよい。
S1〜S3 が想定どおり通れば、アーキテクチャ面の未知リスクはほぼ解消される。

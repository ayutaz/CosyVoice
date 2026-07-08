# DELTA-TTS 再現: Phase 0(事前技術検証・調査)

Phase 1(実装)着手前に潰しておくべき技術リスクの調査・検証記録。
計画本体は `docs/delta_tts_reproduction_plan.md` を参照。

**ステータス: 完了(2026-07-08)。実装ブロッカー S1〜S4 はすべて解消。**

検証プロセス: S1〜S4 を独立エージェントが並列実施し、別の検証エージェントが
コードスパイク(S1/S3)は「スクリプトの批判的レビュー+再実行」、リサーチ(S2/S4)は
「一次ソースを独立に読み直す敵対的検証」を行った。判定: S1/S3/S4 = confirmed、
S2 = partially_confirmed(核心は確定、周辺1点を訂正 → B-2 参照)。

---

## A. 事前調査で確認済みの事項

### A-1. CosyVoice3 checkpoint は手元に揃っている ✅

`pretrained_models/Fun-CosyVoice3-0.5B/` に以下を確認:

- `llm.pt` / `llm.rl.pt`(LLM 2種。論文は RL に言及なし → **`llm.pt`(非RL)をベースにする**)
- `flow.pt`, `hift.pt`(拡散変換では凍結・共用)
- `speech_tokenizer_v3.onnx` / `speech_tokenizer_v3.batch.onnx`(訓練データのトークン抽出に必要)
- `CosyVoice-BlankEN/`(Qwen2 バックボーン、HF形式)

設定値: `speech_token_size=6561`、語彙 6561+200=6761、hidden 896、24層、intermediate 4864、KVヘッド2(GQA、KV次元128)。

### A-2. SSD ブランチの資産はリモートから取り込み可能 ✅

ローカルブランチには無いが `origin/feature/speech-speculative-decoding` に存在:

| ファイル | 流用先 |
|---|---|
| `scripts/prepare_libritts.py` | LibriTTS データ準備(Phase 2) |
| `cosyvoice/bin/train_draft.py` | `train_delta.py` の雛形(訓練ループ・DDP・Windows対応済み) |
| `scripts/eval_ssd.py` | RTF/速度評価スクリプトの雛形 |
| `examples/libritts/cosyvoice3/conf/cosyvoice3_draft.yaml` | `cosyvoice3_delta.yaml` の雛形 |

取り込みは `git checkout origin/feature/speech-speculative-decoding -- <path>`。

### A-3. 推論の統合点は既存の generator インターフェースに載る ✅

`CosyVoice3Model.llm_job`(`cosyvoice/cli/model.py:101`)は `self.llm.inference(...)` の
トークン generator を消費する構造。拡散デコーダは「T ステップ反復の完了後に全トークンを
一括 yield する generator」として実装すれば、`stream=False` 経路に無改造で載る。

### A-4. 評価リソース・参考実装 ✅

- [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval): test-en 1,088件 + WER(Whisper-large-v3)+ SIM(WavLM-large finetuned SV)。論文と同一プロトコル
- 参考実装: [HKUNLP/DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA)(AR→拡散適応の本家、ICLR2025)、[ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)、[Dream 7B](https://hkunlp.github.io/blog/2025/dream/)。DELTA-TTS 公式実装は無し

---

## B. 技術スパイク結果(全4件 PASS、検証済み)

### B-1. S1: 双方向 attention は 4D マスクで実現できる — **PASS / confirmed**

スクリプト: `scripts/spikes/s1_bidirectional_attention.py`(全5チェック PASS。検証エージェントが
無変更で再実行し全数値がビット一致で再現、さらに補完チェック—4D因果構造マスクの対照実験・
sdpa 直接摂動—も追加実施して PASS)。

確認された事実:

- デフォルト(2D マスク)は causal(未来の摂動が過去の hidden に影響 0.0)。**additive 4D float
  マスク `torch.zeros(B,1,T,T)` を渡すと双方向化**(摂動が過去に伝播、max|dh|=64.1)
- コードパス: `_prepare_4d_causal_attention_mask_with_cache_position`(modeling_qwen2.py:698-700)が
  dim==4 のマスクを**そのまま**使う。sdpa 経路で 4D マスクが無視される条件は存在しない
  (modeling_attn_mask_utils.py:286-287)
- パディングとの合成も厳密(キー側列を `finfo(dtype).min` にすれば有効位置は bit 一致で不変)
- eager / sdpa の数値差は相対 ~1e-5(fp32 カーネル累積誤差レベル、実用上問題なし)

Phase 1 実装仕様:

1. マスクは `(B, 1, T_q, T_k)` の additive float。**0.0=可視、`torch.finfo(dtype).min`=遮断**。
   dtype はモデル dtype に一致させる(bf16 学習なら bf16)
2. 呼び出しは既存の `cosyvoice/llm/llm.py:238` と同形:
   `model.model(inputs_embeds=xs, attention_mask=mask4d, use_cache=False)`
3. **4D を渡すと transformers 側の causal・パディング・sliding window 合成はすべてバイパス**される。
   双方向性/パディング等の構造は呼び出し側が 4D マスクに全部エンコードする(2D との自動合成なし)
4. バッチ内可変長は `mask4d[b, :, :, L_b:] = min`。**クエリ側パディング行を「全列遮断」にしない**
   (CPU/sdpa では `_unmask_unattended` が効かず NaN リスク。行は有効キーを見られる状態にしておく)
5. **flash_attention_2 は 4D additive マスク非対応** → H100 学習でも `attn_implementation='sdpa'` を明示
6. `use_cache=False` 必須。position_ids はデフォルト(arange)のままでよい

### B-2. S2: shift operation の定義確定 — **PASS / partially_confirmed(核心は確定)**

一次情報4件(DELTA-TTS 本文 Sec 3.4、DiffuGPT/DiffuLLaMA 論文+訓練/推論コード、Dream 推論コード)
がすべて一致。**「位置 i の hidden が位置 i のトークンを予測」という以前の要約は誤りと確定**。

確定した仕様:

- **入力側は絶対にシフトしない**。[M] 埋め込みはマスクする位置 j 自体に置く
- **シフトは出力側のみ**: AR の契約どおり hidden[i] → llm_decoder → 「トークン i+1」の logits。
  マスク位置 j のトークンは **hidden[j−1] の logits から読む**
- 事前学習済み llm_decoder・speech_embedding・全 Transformer 重みは無変更で再利用でき、
  差分は LoRA が吸収する(これが shift を使う理由。DiffuGPT: シフトなしだと入出力の
  misalignment で適応が困難)

訓練損失(インデックス厳密、DiffuLLaMA train.py と同形):

```python
t    = uniform(0, 1)                                # サンプルごと
mask = (rand(L) < t) & is_target_region             # s_target 領域のみ
x_t  = where(mask, MASK_ID, x_0)                    # [M] は位置 j 自体
logits = llm_decoder(backbone(embed(x_t), bidirectional=True))
sl, tgt, m = logits[:, :-1], x_0[:, 1:], mask[:, 1:]   # 出力側の shift
loss = (1.0 / t) * CE(sl[m], tgt[m])                # 1/t 加重、マスク位置のみ
```

推論(Dream 式の右シフト):

```python
logits = cat([logits[:, :1], logits[:, :-1]], dim=1)   # 位置 j に hidden j-1 の logits
mi   = (x == MASK_ID)
cand = top_p_sample(logits[mi], p=0.8)               # 確信度 = サンプルされたトークンの確率
c_n  = mu*(n/T) / (1 + (mu-1)*(n/T))                 # μ = t_shift = 0.3(強い推定)
k    = floor(c_n*L) - floor(c_prev*L)
if n == T: k = mi.sum()      # 【必須】浮動小数点で c_T<1 になり1トークン残る罠の回避
# 確信度上位 k 個のみ確定、残りは [M] のまま次ステップへ
```

実装上の注意:

- **浮動小数点罠(数値検証で確認済み)**: μ=0.3 で c_T=0.9999999999999998 < 1 となり最終ステップに
  1トークン残る。最終ステップは「残りマスク全部」を強制すること
- 1/t 重みは t→0 で発散 → t のサンプリングに下限クリップ(例 t≥0.01)を入れると安定
- stop token(6561 以降)は logits を −inf にマスクして speech トークンのみサンプル
- 位置0(SOS)は絶対にマスクされないため、右シフト後の logits[:, :1] がダミーでも問題なし

**検証エージェントによる訂正(重要)**: 「論文の主評価は GT 長」という記述は誤り。
Table 1 の主結果(WER 1.75% / SIM 0.688)は ablation の「rule-based length」行と数値が完全一致
しており、**主評価はルールベース長 `r_prompt × W_target` で行われている**(GT 長変種は
1.63% / 0.686)。Phase 1 でもルールベース長をそのまま標準採用してよい。

### B-3. S3: peft LoRA — **PASS(7/7)/ confirmed、rank r=64 が確定**

スクリプト: `scripts/spikes/s3_lora_peft.py`(検証エージェントが再実行し 7/7 PASS、
全数値再現、空虚チェックの疑い精査も問題なし)。

確認された事実:

- `LoraConfig(r=64, lora_alpha=128, target_modules=[q,k,v,o,gate,up,down], lora_dropout=0.0, bias='none')`
  で **trainable = 35,192,832 と逆算値に厳密一致 → 論文の +35M は r=64 で確定**
- 注入モジュール 168 個(24層×7)。lm_head / embed_tokens への注入なし
- `get_peft_model` はサブモジュールを**インプレース置換**するため、`Qwen2Encoder` の
  直呼び経路(`self.model.model(...)`)でも LoRA が有効 — **llm.py の forward 変更は不要**
- `disable_adapter()` も直呼び経路に効く(LoraLayer 自体のフラグ切替のため)
- save/load ラウンドトリップは bit-exact。adapter_model.safetensors = 140.8MB(fp32。bf16 化で半減可)
- backbone は全凍結(requires_grad=True は lora_* の 336 テンソルのみ)

Phase 1 実装仕様:

1. `task_type=None` でよい(generate() を使わずバックボーン直呼びのため)
2. `get_peft_model` の戻り値(PeftModel)への参照を必ず保持(save/disable/merge 用)
3. `PeftModel.from_pretrained` はデフォルト推論用(requires_grad=False)。**学習再開時は
   `is_trainable=True`** を渡す
4. checkpoint は full state_dict に peft 命名が混ざるのを避け、**adapter を `save_pretrained` で分離保存**
5. lora_dropout は論文未記載 → 0.0 で開始(要調整項目)

### B-4. S4: 訓練時 prompt/target 分割 — **PASS / confirmed、方式決定**

一次情報8件(DELTA-TTS、MaskGCT、SoundStorm、VALL-E、E2-TTS、F5-TTS、LLaDA-TTS、CosyVoice2)を
精査。**DELTA-TTS 本文に訓練時の prompt 構成の記載が無いことを原文確認**した上で、先行研究の
確立手法から設計を決定した。

**採用: 方式(a)変形 = 同一発話接頭辞 + テキスト非分割**(MaskGCT T2S が文字通り採用する確立手法):

1. speech トークン列の接頭辞を s_prompt に: `L = floor(u·T), u ~ U(0, 0.5)`
   (`--prompt_ratio_max` フラグで可変)。s_prompt は常時可視・損失対象外
2. **テキストは分割しない**: t_prompt = 空、t_target = 発話全文の転写。
   **音声-テキストアライメントは一切不要**(これが (a) の唯一の懸念だったが、
   テキスト非分割で完全に回避できることが MaskGCT で実証済み)
3. **推論時も対称に**: プロンプト転写は t_prompt に入れず **t_target の先頭に連結**
   (単一テキスト領域規約 = MaskGCT/E2/F5/CosyVoice2-ICL と同一。CosyVoice バックボーンの
   事前学習分布とも整合)。訓練 t_prompt 空 / 推論非空の非対称は分布外なので禁止
4. プロンプトドロップ: 確率 0.1〜0.2 で L=0 の標本を混ぜる(頑健性 + 将来の CFG 用)
5. マスクは s_target のみ、iid Bernoulli(t), t~U(0,1)。推論初手(s_prompt 全可視 + s_target 全マスク)は
   t→1 の訓練標本と同分布
6. フラグ設計: `--prompt_mode {same_prefix (default), other_utterance, none}`。
   other_utterance(同一話者別発話、t_prompt 非退化)は ablation 用。
   **none は話者クローンが原理的に不成立**(推論初手で可視トークンが無く話者情報の入力経路がない)
   のでベースライン確認専用
7. 推論時の target 長: ルールベース `r_prompt × W_target`(論文の主評価設定と同じ)
8. t_inst の中身は論文未記載 → 固定文字列(または空)で訓練・推論同一に

根拠の要点: 成功しているマスク生成系 TTS は**全て**訓練時に連続可視のプロンプト領域を明示的に
確保している(MaskGCT: 接頭辞切り出し+テキスト全文条件 / SoundStorm: 非マスク接頭辞を必ず
サンプル / VALL-E NAR: 同一発話3秒 / E2・F5: 連続スパンマスクで残部が自然にプロンプト化)。
純 iid マスクのみで接頭辞条件付けを訓練した成功例は確認されなかった。VALL-E は同一発話接頭辞
訓練 → 別発話プロンプト推論への汎化を直接実証しており、(a) で訓練して CosyVoice 形式の
ゼロショット推論を行う構成は妥当。

---

## C. 残る未解決事項(Phase 1 で対応)

| 項目 | 内容 | 対応 |
|---|---|---|
| 総訓練ステップ数 | 論文未記載 | 損失と検証 WER で決定 |
| t のサンプリング分布 | 「t∈(0,1)」のみ(一様と推定) | 一様 + 下限クリップで開始 |
| lora_dropout | 未記載 | 0.0 で開始 |
| 接頭辞率の上限 | U(0,0.3) vs U(0,0.5) vs 全域 | 小規模 ablation |
| conv モジュールの挿入位置の詳細 | attention 後か FFN 後か(「各ブロック後に残差」とのみ) | Appendix A.1 を実装時に精読 |
| H100/bf16 での sdpa パリティと学習速度 | スパイクは CPU/fp32 | Phase 2 冒頭で確認。遅ければ FlexAttention 検討 |
| μ=t_shift の同一性 | 強い推定(式にパラメータが μ 1つ、レシピに t_shift=0.3 のみ) | 実装は同一として進め、挙動で確認 |

## D. GPU 検証(Phase 2 前に実施)

- **S5: AR ベースライン RTF 計測** — H100 で end-to-end / LM 単体、音声長ビン別。
  論文値(A100)とは絶対値が異なるため speedup 比で比較
- **S6: LibriTTS パイプライン疎通** — prepare_libritts.py の CV3(tokenizer v3)対応、
  585h 抽出のコスト見積もり
- **S7: 長尺一括合成確認** — stream=False で 10 秒級の一括 token2wav の品質・VRAM

## E. 判定

**Phase 1 実装ブロッカーはすべて解消。** アーキテクチャ面の未知リスク(双方向化・LoRA・shift・
prompt 構成)は検証済みの実装仕様に落ちており、実装に着手できる。依存は `peft==0.19.1` を
`uv add` 済み(transformers 4.51.3 と互換)。

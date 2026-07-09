# DELTA-TTS 再現: Phase 1 実装ドキュメント

Phase 1(拡散版 LM・訓練スクリプト・推論デコーディングの実装)の記録。2026-07-09 完了。
仕様の根拠は `docs/delta_tts_phase0_verification.md` §B(Phase 0 で確定した実装仕様)。

## 1. 成果物

| ファイル | 内容 |
|---|---|
| `cosyvoice/llm/diffusion_llm.py` | コア実装(下記 §2) |
| `cosyvoice/bin/train_delta.py` | 訓練エントリポイント(§3) |
| `examples/moe_speech/cosyvoice3/conf/cosyvoice3_delta.yaml` | **訓練設定(日本語 moe-speech-plus)** |
| `tests/test_delta_llm.py` | ユニットテスト 48 件(§5) |
| `scripts/spikes/s8_delta_smoke.py` | 0.5B 実機スモーク(再実行可能) |

**既存ファイルは一切変更していない**(git diff で確認済み)。

## 2. コア実装(`cosyvoice/llm/diffusion_llm.py`)

### クラス構成

- **`DiffusionCosyVoice3LM(CosyVoice3LM)`** — 拡散版 LM 本体
  - `forward(batch, device)`: masked diffusion 訓練損失。系列 `[sos, (instruct,) text(全文・非分割), task, s_prompt(可視), s_target(マスク), eos]`、
    プロンプト接頭辞 `L~U(0, prompt_ratio_max·T)`(確率 `prompt_drop` で 0)、`t~U(t_min,1)` の iid マスク、
    1/t 加重 CE(マスク位置のみ、出力側 shift)。batch に `instruct_token` があれば text 領域の
    先頭に連結(可視・損失対象外。日本語 FT の学習分布および論文の t_inst と整合)
  - `inference_diffusion(...)`: 信頼度順序デコーディング。Qwen2LM.inference 互換の generator
    (デコード完了後にトークンを逐次 yield → `cli/model.py` の `llm_job` と互換)
  - `apply_lora()` / `attach_conv_modules()` / `init_mask_embedding()` / `freeze_for_delta()` /
    `trainable_parameter_summary()`: AR→拡散変換の各ステップ
  - `inference()` / `inference_bistream()`: **NotImplementedError でAR経路を明示的に封鎖**
- **`ConformerConvModule`** — LN → pw(d→2d) → GLU → depthwise(k=31) → LN → SiLU → pw(d→d) → Dropout → 残差。
  最終 pw は**ゼロ初期化**(変換直後は恒等 = 事前学習挙動を保存)。pad_mask でパディングリーク防止
- 純関数(単体テスト可能): `unmask_schedule(L,T,mu)`(sum==L 保証・最終ステップ強制)、
  `shift_align(logits,targets,mask)`(出力側 shift の整列)、`build_bidirectional_mask(pad_mask,dtype)`(4D additive)

### 変換手順(順序が重要)

```
1. DiffusionCosyVoice3LM を構築(yaml の llm セクション)
2. llm.pt を strict=False でロード   ← attach 前に行う(下記の注意)
3. apply_lora()                       ← peft r=64/α=128、PeftModel 参照を保持
4. attach_conv_modules()              ← decoder layer を ConvWrappedDecoderLayer で包む
5. init_mask_embedding()              ← speech_embedding[:6561] の平均で [M] を初期化
6. freeze_for_delta()                 ← LoRA + conv + mask_emb のみ trainable
```

**注意**: `attach_conv_modules()` は backbone の state_dict キーを `layers.N.*` → `layers.N.layer.*` に
1段ネストさせる。**チェックポイントのロードは必ず attach 前に行う**(train_delta.py はこの順序を実装済み。
変換後 checkpoint を `--checkpoint` に誤って渡すと RuntimeError で拒否するガード付き)。

### 0.5B 実機での検証値(s8 スモーク、CPU/fp32)

- llm.pt ロード: missing=`['mask_emb']` のみ、unexpected=0 ✅
- trainable: **lora=35,192,832(論文の 35M と厳密一致)、conv=58,641,408(≈59M)、mask_emb=896、other=0** ✅
- forward: 合成バッチで損失有限(raw CE=10.73 ≒ 一様分布エントロピー ln(6761)=8.82 の想定域)✅
- デコード: T=4, target_len=25, schedule=[2,3,6,14]、25トークン全て [0,6561) ✅(CPU ~1.2s/step)

## 3. 訓練(`train_delta.py` + `cosyvoice3_delta.yaml`)

凍結バックボーンは日本語 FT 済み llm、データは moe_speech parquet:

```bash
PYTHONPATH=third_party/Matcha-TTS:. uv run python cosyvoice/bin/train_delta.py \
    --train_engine torch_ddp --ddp.dist_backend gloo --model llm \
    --config examples/moe_speech/cosyvoice3/conf/cosyvoice3_delta.yaml \
    --train_data data/moe_speech/train.data.list --cv_data data/moe_speech/dev.data.list \
    --model_dir ./checkpoints_delta_ja \
    --checkpoint checkpoints/cosyvoice3_ja/llm.pt \
    --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN
```

※ `checkpoints/cosyvoice3_ja/llm.pt` は save_model 形式(`epoch`/`step` キー入り)だが、
build_delta_model は AR checkpoint から訓練位置を継承しない(unexpected キーとして無害にスキップ)。
parquet リストは `examples/moe_speech/cosyvoice3/run.sh` の stage で生成(HF gated 認証が必要)。

※ 英語(LibriTTS)用の設定 `examples/libritts/cosyvoice3/conf/cosyvoice3_delta.yaml` は
2026-07-09 のスコープ変更(英語再現は対象外)で削除。必要になれば git 履歴 `40f9e32` から復元可能。

- **スケジューラ**: 論文の「warmup 2000 → lr=1e-4 定数」を再現するため、train_delta.py 内に
  `ConstantWithWarmupLR` を定義し、`init_optimizer_and_scheduler` が返す WarmupLR
  (warmup 後 inverse-sqrt 減衰 → 論文と乖離)を差し替える。yaml を意図的に
  NoamHoldAnnealing 等へ変えた場合は差し替えない
- バッチ: static batch で論文の実効バッチ 16(発話)を厳守(accum × batch の積を 16 に保つ)
- resume は `--delta_checkpoint`(AR の `--checkpoint` からは step/epoch を継承しない)。
  チェックポイントは full state_dict と trainable のみの分離保存の両方
- CPU でも訓練ステップが動く(forward 内で device を正規化。GPU 挙動は不変)

## 4. 推論の暫定仕様(Phase 2 で CLI 統合予定)

- ルールベース長: `target_len = ceil(len(prompt_speech)/len(prompt_text_tokens) × len(text_tokens) × length_scale)`、
  `max_token_text_ratio` で上限クランプ。**トークン数比で代用中**(論文は文字数基準。CLI 統合時に置換)
- プロンプト片側のみ空の場合はフォールバック定数比 6.0(docstring に明記)
- `cli/model.py` / `cli/cosyvoice.py` への配線は未実施(Phase 2)。AR 経路を誤って呼ぶと
  明示的な NotImplementedError

## 5. 品質保証

- **テスト**: `tests/test_delta_llm.py` 48 件(ミニ Qwen2 構成、0.5B 非依存)。スケジュール(浮動小数点罠含む)、
  shift 整列、マスキング/レイアウト、instruct 領域の連結、4D マスク双方向性、conv 恒等・リーク・パラメータ数、
  凍結範囲、デコード、forward+backward 勾配経路、バッチ内パディング隔離(統合不変量)、ConstantWithWarmupLR。
  **全スイート 85 passed(既存 37 件にリグレッションなし)**
- **多段検証**(ultracode): 実装 → テスト(初回 40 件 green、実装修正ゼロ)→ 3方向レビュー
  (仕様忠実性/正当性/統合)で 13 指摘 → 敵対的検証で 12 confirmed / 1 refuted → 全 confirmed を修正適用。
  主な修正: スケジューラ差し替え(major)、k_n=0 ステップの無駄 forward 除去(RTF 計測の公平性)、
  変換後 checkpoint 誤投入ガード、AR checkpoint からの step/epoch 誤継承除去
- refuted された指摘: 「bf16 AMP 時に 4D マスクの dtype が不一致」→ 実装は入力埋め込みの dtype に
  合わせてマスクを構築しており問題なし

## 6. 既知の制約・Phase 2 への引き継ぎ

1. 訓練前の出力は縮退(LoRA-B/conv ゼロ初期化のため)— スモークは機構の検証のみ
2. 1/t 加重損失は小バッチで高分散(実測: weighted 86.3 vs raw CE 10.7)→ 訓練時は生 CE も併記ログ推奨
3. instruct(t_inst)対応済み: batch の `instruct_token` を text 領域へ連結(日本語 FT は全発話
   `You are a helpful assistant.<|endofprompt|>` 付きで学習されているため必須)。推論では instruct は
   text トークン列の中に入って届く(AR の cross-lingual 経路と同じ)ため専用引数は不要
4. 訓練データ準備は Phase 2: moe_speech parquet(vast.ai で run.sh 再実行 or 前回シャード再利用)
5. 'Sliding Window Attention is enabled but not implemented' 警告は既存 AR ロードと同じで無害

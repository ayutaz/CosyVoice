# DELTA-TTS 再現: Phase 2(日本語 delta 訓練)結果

2026-07-09 実施。vast.ai H100 で日本語 FT 済み CosyVoice3 を凍結バックボーンとして delta 変換を訓練し、
ローカル RTX 4070 Ti SUPER で CER / RTF を評価した。

## 1. 結果サマリ

`scripts/eval_ja_cer.py`、漢字混じり20文、whisper large-v3、全経路を同一マシン・同一条件で新規合成:

| 経路 | avg CER | avg RTF | 速度比 |
|---|---|---|---|
| base_katakana(事前学習 AR + カタカナ変換) | 0.0968 | 1.84 | 1.0× |
| ft_kanji(日本語 FT AR + 漢字直接)= 従来最良 | 0.0854 | 1.89 | 1.0× |
| delta 単体 step 25,000 | 0.1378 | 1.11 | 1.7× |
| **delta 平均化(20k+25k+30k)= 最終成果物** | **0.1058** | **0.945** | **2.0×** |

- **delta はカタカナ変換 AR 経路(0.097)とほぼ同等の CER を、エンドツーエンド2倍速で達成**
- 目標(CER ≤ 0.10、3×速)にはわずかに未達。改善レバーは §5
- RTF はエンドツーエンド値(flow/hift の固定コストを含む)。論文の 3.3× は A100 + 共有部の
  最適化込みの値であり、LM 単体の高速化はこれより大きい
- 最終チェックポイント: `checkpoints/delta_ja/avg_20k30k_delta.pt`
  (`scripts/average_delta_checkpoints.py` で 20k/25k/30k を平均)

## 2. 訓練の経過と重要な所見

- 設定: `examples/moe_speech/cosyvoice3/conf/cosyvoice3_delta.yaml`(論文レシピ: AdamW、
  warmup 2000 → 定数 lr 1e-4、実効バッチ16、bf16)、backbone = `checkpoints/cosyvoice3_ja/llm.pt`
- データ: moe-speech-plus 289,893 発話(前回 AR FT と同一の 470h、MOS/cross-CER フィルタ済み)
- **品質は step 20k〜30k(エポック1〜1.5)でピーク、以降は過学習で劣化**:

| step | 5k | 10k | 15k | 20k | 25k | 30k | 40k | 50k |
|---|---|---|---|---|---|---|---|---|
| CER(5文) | 0.58 | 0.28 | 0.71* | 0.20 | 0.062 | 0.21 | 0.23 | 0.35 |

  CV loss も同様(30k: 6.68 → 50k: 6.93)。step 54,000 で手動停止(予定5エポック=90k の前)。
  *個々のチェックポイントは定数 lr のため品質が大きく振動する(15k の外れ値等)→ 平均化が有効
- inf grad_norm がステップの ~0.5% で発生(1/t 加重の裾)。当該ステップはスキップされ実害なし
- 訓練スループット: 0.186 s/step(H100、バッチ16、dataloader 24 workers)

## 3. インフラ・運用記録(vast.ai)

- インスタンス: H100 SXM 80GB / 31コアquota / 1.2TB / $2.20/hr(id 44269445、**破棄済み**)
- 総費用: **~$13**(DL 51分 + 前処理 ~1.7h + 訓練 ~2.5h + 回収)
- 踏んだ問題と対処(すべて `scripts/prod/vast_delta_run.sh` に反映済み):
  1. nvidia/cuda:12.8 イメージの compat libcuda がホストドライバ(555/CUDA 12.5)と衝突し
     **CUDA error 803**(torch も ORT も全滅)→ compat 無効化で解決
  2. onnxruntime-gpu が cudnn を見つけられず **CPU フォールバック**(トークン抽出が10倍遅い)
     → torch 同梱 nvidia libs を LD_LIBRARY_PATH に追加
  3. `torchrun` は venv 外から呼べない → `uv run torchrun`
  4. 31コア quota では dataloader 8 workers で GPU 飢餓(0.9s/step)→ 24 workers で 0.19s/step
- **HF private への大容量アップロードは LFS 403 で失敗**(小ファイルのコミットは成功)。
  アカウントのプライベートストレージ枠か fine-grained トークンの権限が原因の可能性。未解決。
  代替としてローカル回収で対応

## 4. 成果物(ローカル `checkpoints/delta_ja/`、git 管理外)

| ファイル | 内容 |
|---|---|
| `avg_20k30k_delta.pt` | **最終 delta(平均化)** — `from_ar(..., delta_checkpoint=これ)` で使用 |
| `epoch_*_step_*_delta.pt` × 8 | ピーク周辺の単体チェックポイント(5k〜50k) |
| `delta_data_artifacts.tgz` | **波形フリー parquet 一式(523MB)+ tensorboard + yaml**。次回の訓練で 326GB DL と 2.5h 前処理をスキップできる |

評価音声: `eval_out_delta_final/`(3経路×20文)、`eval_out_delta_avg20/`(平均化版)、report.json 付き。

## 5. 次の改善候補(優先度順)と実施状況

1. **低 lr での短い追い込み** → **実行済み**(lr 1e-5、25k から1エポック。結果は
   `docs/delta_tts_speed_quality_report.md` に追記)
2. **推論パラメータ探索(T スイープ)** → **実行済み。T=8 が最適: CER 0.0908 / E2E 2.52× /
   LM 5.27×(目標 CER≤0.10 達成)**。詳細計測は `docs/delta_tts_speed_quality_report.md`
3. 平均化の窓の最適化(refine 後の 500 ステップ刻みチェックポイントで SWA 窓探索)
4. RTF 深掘り: fp16 推論、flow/hift 側の高速化(AR と共通なので speedup 比は不変だが
   絶対 RTF が下がる)
5. 音素バランスの良い訓練データ拡張(8000h 計画と合流)

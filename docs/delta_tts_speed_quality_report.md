# DELTA-TTS 日本語適用: 速度・品質計測レポート

2026-07-09 計測。日本語 FT 済み CosyVoice3 の AR デコーディングと、DELTA-TTS 変換
(masked diffusion、`checkpoints/delta_ja/avg_20k30k_delta.pt`)の拡散デコーディングを、
同一マシン・同一文・同一シードで比較した。

## 1. 結論

**DELTA-TTS の日本語適用により、品質をほぼ維持したまま(CER +0.005)、LM を 5.3倍、
エンドツーエンドで 2.5倍高速化できた。** 拡散ステップ数は論文既定の T=16 ではなく
**T=8 が最適**(CER・速度の両方で T=16 を上回る)。

| 経路 | CER ↓ | 完全一致 ↑ | SIM ↑ | RTF ↓ | LM-RTF ↓ | E2E 速度比 | LM 速度比 |
|---|---|---|---|---|---|---|---|
| AR(ft_kanji、従来) | **0.0854** | 9/20 | 0.760 | 1.850 | 1.319 | 1.0× | 1.0× |
| delta T=16(論文既定) | 0.1058 | 7/20 | 0.752 | 0.904 | 0.438 | 2.05× | 3.01× |
| **delta T=8(推奨)** | **0.0908** | **9/20** | 0.754 | **0.734** | **0.250** | **2.52×** | **5.27×** |
| delta T=4 | 0.2266 | 0/20 | 0.739 | 0.622 | 0.136 | 2.97× | 9.66× |

- **T=8 は目標(CER ≤ 0.10)を達成**し、完全一致文数(CER=0 の文)は AR と同数の 9/20
- T=4 は破綻(CER 0.23、完全一致 0)— 信頼度順序デコーディングには最低限の反復が必要
- 話者類似度(campplus cosine、プロンプト vs 生成音声)は全経路で ~0.75 と AR(0.760)から
  ほぼ劣化なし
- 音声長ビン別の E2E speedup(T=8): 0-4秒文 2.57× / 4-6秒文 2.44×(論文表3と同傾向で
  長さ依存は小さい。paper は 5-10s でより大きな speedup を報告しているが、本評価文は
  最大 ~6 秒のため長尺域は未計測)

## 2. 計測方法

- スクリプト: `scripts/eval_delta_speed_quality.py`(report.json: `eval_out_speed_quality/`)
- ハードウェア: RTX 4070 Ti SUPER(ローカル)、fp32、`stream=False`
- 文セット: `eval_ja_cer.py` の漢字混じり20文(学習データ外、同形異音語・助数詞・日付・数値)
- プロトコル:
  - 全経路とも `inference_cross_lingual`(instruct + 漢字直接入力、text_frontend=False)、
    文ごとに同一シード
  - **各経路の計測前にウォームアップ合成1回**(初回 CUDA ウォームアップが AR 側を
    ~40% 不利にするのを排除)
  - **LM-RTF**: llm_job 内のトークン generator 消費時間のみを計測(flow/hift を除外)。
    stream=False では LM → flow が直列のため清潔に分離できる
  - CER: whisper large-v3(language=ja, temperature=0)、NFKC + 記号除去後の編集距離
  - SIM: campplus(学習時と同一の話者埋め込みモデル)による cosine 類似度
- delta モデル: `avg_20k30k_delta.pt`(step 20k/25k/30k の SWA 平均。単体 25k は CER 0.138 で
  平均化が必須 — `docs/delta_tts_phase2_results.md` §2)

## 3. 論文との対応

| 指標 | 論文(英語、A100、T=16) | 本計測(日本語、4070 Ti) |
|---|---|---|
| 品質 | WER 1.75%(AR 2.02% より**改善**) | CER 0.0908(AR 0.0854 より +0.005 **僅かに劣化**) |
| E2E 速度比 | 3.3× | 2.05×(T=16)/ 2.52×(T=8) |
| LM 速度比 | (未報告) | 3.01×(T=16)/ **5.27×(T=8)** |

- E2E 速度比が論文より小さいのは flow/hift の固定コストの比率差(本環境では delta T=8 の
  RTF 0.734 のうち LM は 0.250 のみで、残り ~0.48 は AR と共通の flow/hift)。LM 単体では
  論文の全体値を上回る高速化が出ており、変換手法自体は日本語でも論文どおり機能している
- 品質が AR を上回れなかった点は論文と異なる(訓練データ量 470h vs 585h、バックボーンが
  既に日本語 FT 済みで AR side が強い、訓練レシピの総ステップ数が非公開、等の差)

## 4. 実行時条件・再現手順

```bash
# 計測の再現(~30分、ローカル GPU)
.venv/Scripts/python.exe scripts/eval_delta_speed_quality.py \
    --delta_checkpoint checkpoints/delta_ja/avg_20k30k_delta.pt \
    --num_steps 16 8 4 --out_dir eval_out_speed_quality
```

- 推論時に T を変える場合は `DiffusionCosyVoice3LM` の `num_steps` 属性を設定
  (`from_ar(..., num_steps=8)` または実行中に `model.llm.num_steps = 8`)
- 絶対 RTF はハードウェア・精度(fp32)依存。速度比は同一条件の相対値

## 5. 既知の限界と次の計測

1. 文セットは20文・単一プロンプト話者。より広い話者・文長分布(特に 5-10 秒超)での
   検証は今後の課題
2. MOS(自然性)は未計測(speechmos 未導入)。CER/SIM は明瞭性・話者性の代理指標
3. **refine 訓練(lr 1e-5 で 25k から追い込み)を実行中** — 完了後、最良チェックポイントで
   本ベンチマークを再実行し、本レポートに追記予定(T=8 で AR 超えが目標)

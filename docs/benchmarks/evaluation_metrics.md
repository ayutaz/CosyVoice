# 日本語TTS評価指標ガイド

## 目次

1. [概要](#概要)
2. [客観評価指標](#客観評価指標)
3. [主観評価指標](#主観評価指標)
4. [評価スクリプト](#評価スクリプト)
5. [ベースライン設定](#ベースライン設定)

## 概要

日本語TTSシステムの品質評価には、客観的指標と主観的指標の両方が必要です。本ガイドでは、CosyVoice 2.0の日本語改善効果を測定するための評価手法を提供します。

## 客観評価指標

### 1. WER (Word Error Rate)

**定義**: ASRで認識したテキストと正解テキストの単語誤り率

**計算式**:
```
WER = (S + D + I) / N
S: 置換エラー数
D: 削除エラー数
I: 挿入エラー数
N: 正解単語数
```

**実装**:
```python
# tools/evaluate_wer.py
from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

def evaluate_wer(model_path, test_dataset):
    """
    日本語ASRでWER評価

    Args:
        model_path: CosyVoiceモデルパス
        test_dataset: テストデータ（text, audioのペア）

    Returns:
        WER score
    """
    # 日本語ASRモデル
    processor = Wav2Vec2Processor.from_pretrained("rinna/japanese-wav2vec2")
    asr_model = Wav2Vec2ForCTC.from_pretrained("rinna/japanese-wav2vec2")
    asr_model.eval()

    # CosyVoiceモデル
    from cosyvoice.cli.cosyvoice import CosyVoice2
    tts_model = CosyVoice2(model_path)

    references = []
    hypotheses = []

    for sample in test_dataset:
        # TTS合成
        synth_audio = synthesize_audio(tts_model, sample['text'])

        # ASR認識
        inputs = processor(synth_audio, sampling_rate=24000, return_tensors="pt")
        with torch.no_grad():
            logits = asr_model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        references.append(sample['text'])
        hypotheses.append(transcription)

    wer_score = wer(references, hypotheses)
    return wer_score
```

**目標値**:
- Baseline: 25-30%
- Phase 1後: 20-22%
- Phase 2後: 18-20%
- Phase 3後: 15%以下

### 2. CER (Character Error Rate)

**定義**: 文字レベルの誤り率（日本語では単語分割が曖昧なためWERより有用）

**実装**:
```python
from jiwer import cer

def evaluate_cer(references, hypotheses):
    """文字誤り率の計算"""
    cer_score = cer(references, hypotheses)
    return cer_score
```

**目標値**:
- Baseline: 12-15%
- Phase 1後: 10-11%
- Phase 2後: 8-9%
- Phase 3後: 6%以下

### 3. アクセント精度

**定義**: pyopenjtalkで推定した正解アクセントと音声から抽出したアクセントの一致率

**実装**:
```python
# tools/evaluate_accent.py
import pyopenjtalk
import librosa
import numpy as np

def extract_accent_from_text(text):
    """テキストからアクセント型を抽出"""
    labels = pyopenjtalk.extract_fullcontext(text)
    accents = []

    for label in labels:
        accent_info = parse_accent_info(label)
        if accent_info:
            accents.append(accent_info)

    return accents

def extract_accent_from_audio(audio, sr=24000):
    """音声からF0軌跡を抽出してアクセント推定"""
    # F0抽出
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )

    # F0の変化パターンからアクセント型を推定
    accent_pattern = analyze_f0_pattern(f0)
    return accent_pattern

def analyze_f0_pattern(f0):
    """F0変化からアクセント型を推定"""
    # NaNを除去
    f0_clean = f0[~np.isnan(f0)]

    if len(f0_clean) < 2:
        return 0  # アクセント型なし

    # F0の変化点検出
    diff = np.diff(f0_clean)
    threshold = np.std(diff) * 0.5

    # 下降点を探す（アクセント核）
    fall_points = np.where(diff < -threshold)[0]

    if len(fall_points) == 0:
        return 0  # 平板型
    else:
        # 最初の下降点の位置
        accent_position = fall_points[0] + 1
        return accent_position

def evaluate_accent_accuracy(model_path, test_dataset):
    """アクセント精度評価"""
    from cosyvoice.cli.cosyvoice import CosyVoice2
    tts_model = CosyVoice2(model_path)

    correct = 0
    total = 0

    for sample in test_dataset:
        # 正解アクセント
        true_accents = extract_accent_from_text(sample['text'])

        # 合成音声
        synth_audio = synthesize_audio(tts_model, sample['text'])

        # 音声からアクセント推定
        pred_accents = extract_accent_from_audio(synth_audio)

        # 比較（簡易版、実際にはDTWでアライメント必要）
        if compare_accent_patterns(true_accents, pred_accents):
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy
```

**目標値**:
- Baseline: 65-70%
- Phase 1後: 80-85%
- Phase 2後: 90-92%
- Phase 3後: 95%+

### 4. MCD (Mel-Cepstral Distortion)

**定義**: 合成音声と真の音声のメルケプストラム距離

**実装**:
```python
import librosa
import numpy as np
from scipy.spatial.distance import euclidean

def compute_mcd(synth_audio, true_audio, sr=24000):
    """MCD計算"""
    # メルケプストラム抽出
    synth_mfcc = librosa.feature.mfcc(y=synth_audio, sr=sr, n_mfcc=13)
    true_mfcc = librosa.feature.mfcc(y=true_audio, sr=sr, n_mfcc=13)

    # DTWでアライメント
    from fastdtw import fastdtw
    distance, path = fastdtw(synth_mfcc.T, true_mfcc.T, dist=euclidean)

    # MCD計算
    mcd = (10.0 / np.log(10)) * np.sqrt(2 * distance / len(path))
    return mcd
```

**目標値**:
- 優秀: < 4.0 dB
- 良: 4.0-5.5 dB
- 普通: 5.5-7.0 dB
- 要改善: > 7.0 dB

### 5. Prosody Score

**定義**: F0軌跡の相関係数（韻律の自然性）

**実装**:
```python
from scipy.stats import pearsonr

def evaluate_prosody_score(model_path, test_dataset):
    """韻律スコア評価"""
    from cosyvoice.cli.cosyvoice import CosyVoice2
    tts_model = CosyVoice2(model_path)

    correlations = []

    for sample in test_dataset:
        # 真の音声のF0
        true_f0, _, _ = librosa.pyin(sample['audio'], fmin=80, fmax=400, sr=24000)

        # 合成音声のF0
        synth_audio = synthesize_audio(tts_model, sample['text'])
        synth_f0, _, _ = librosa.pyin(synth_audio, fmin=80, fmax=400, sr=24000)

        # DTWアライメント
        aligned_true, aligned_synth = dtw_align_f0(true_f0, synth_f0)

        # 相関係数
        if len(aligned_true) > 0 and len(aligned_synth) > 0:
            corr, _ = pearsonr(aligned_true, aligned_synth)
            correlations.append(corr)

    avg_correlation = np.mean(correlations)
    return avg_correlation
```

**目標値**:
- 優秀: > 0.85
- 良: 0.75-0.85
- 普通: 0.65-0.75
- 要改善: < 0.65

## 主観評価指標

### 1. MOS (Mean Opinion Score)

**定義**: 1-5点スケールでの自然性評価

**評価プロトコル**:
```
評価基準:
5: 非常に自然（人間の音声と区別できない）
4: 自然（ほぼ人間らしい）
3: 普通（合成音声だが理解可能）
2: やや不自然（ロボット的だが内容は理解可能）
1: 非常に不自然（理解困難）
```

**実装**:
```python
# tools/mos_evaluation.py
import random
import pandas as pd

def create_mos_test(models, test_samples, output_dir):
    """
    MOSテスト用のサンプル生成

    Args:
        models: {'model_name': model_path}
        test_samples: テストサンプルリスト
        output_dir: 出力ディレクトリ
    """
    samples = []

    for i, sample in enumerate(test_samples):
        for model_name, model_path in models.items():
            # 音声合成
            audio = synthesize_audio(model_path, sample['text'])

            # ランダムID（ブラインドテスト）
            sample_id = f"mos_{i:03d}_{random.randint(1000, 9999)}"

            # 保存
            save_audio(f"{output_dir}/{sample_id}.wav", audio)

            samples.append({
                'sample_id': sample_id,
                'model': model_name,
                'text': sample['text'],
                'category': sample.get('category', 'general')
            })

    # 評価シート作成
    df = pd.DataFrame(samples)
    df['naturalness'] = ''
    df['prosody'] = ''
    df['clarity'] = ''
    df['overall'] = ''
    df.to_csv(f"{output_dir}/mos_evaluation_sheet.csv", index=False)

    return samples

def analyze_mos_results(results_csv):
    """MOS結果の統計分析"""
    df = pd.read_csv(results_csv)

    # モデルごとの平均MOS
    mos_by_model = df.groupby('model')['overall'].mean()

    # カテゴリごとの分析
    mos_by_category = df.groupby(['model', 'category'])['overall'].mean()

    # 信頼区間
    from scipy import stats
    confidence_intervals = {}
    for model in df['model'].unique():
        scores = df[df['model'] == model]['overall'].dropna()
        ci = stats.t.interval(0.95, len(scores)-1,
                             loc=scores.mean(),
                             scale=stats.sem(scores))
        confidence_intervals[model] = ci

    return {
        'mos_by_model': mos_by_model,
        'mos_by_category': mos_by_category,
        'confidence_intervals': confidence_intervals
    }
```

**目標値**:
- 人間レベル: 4.38-4.5
- 優秀: 4.2-4.4
- 良: 3.8-4.2
- 普通: 3.5-3.8
- 要改善: < 3.5

### 2. CMOS (Comparison MOS)

**定義**: ペア比較による相対評価（-3 to +3スケール）

**実装**:
```python
def create_cmos_test(baseline_model, test_models, test_samples, output_dir):
    """CMOSテスト用のペア生成"""
    pairs = []

    for i, sample in enumerate(test_samples):
        # ベースライン音声
        baseline_audio = synthesize_audio(baseline_model, sample['text'])

        for model_name, model_path in test_models.items():
            # テストモデル音声
            test_audio = synthesize_audio(model_path, sample['text'])

            # ランダムにA/B順序を決定
            if random.random() < 0.5:
                audio_a, audio_b = baseline_audio, test_audio
                label_a, label_b = 'baseline', model_name
            else:
                audio_a, audio_b = test_audio, baseline_audio
                label_a, label_b = model_name, 'baseline'

            pair_id = f"cmos_{i:03d}_{random.randint(1000, 9999)}"

            # 保存
            save_audio(f"{output_dir}/{pair_id}_A.wav", audio_a)
            save_audio(f"{output_dir}/{pair_id}_B.wav", audio_b)

            pairs.append({
                'pair_id': pair_id,
                'text': sample['text'],
                'audio_a_model': label_a,
                'audio_b_model': label_b,
                'score': ''  # -3 to +3
            })

    df = pd.DataFrame(pairs)
    df.to_csv(f"{output_dir}/cmos_evaluation_sheet.csv", index=False)

    return pairs
```

**評価基準**:
```
-3: Bが非常に悪い
-2: Bが悪い
-1: Bがやや悪い
 0: 同等
+1: Bがやや良い
+2: Bが良い
+3: Bが非常に良い
```

### 3. ABX Test

**定義**: 3サンプル（A, B, X）を提示し、XがAとBのどちらに近いか判定

**実装**:
```python
def create_abx_test(model1, model2, ground_truth, test_samples, output_dir):
    """ABXテスト生成"""
    tests = []

    for i, sample in enumerate(test_samples):
        # A: モデル1
        audio_a = synthesize_audio(model1, sample['text'])

        # B: モデル2
        audio_b = synthesize_audio(model2, sample['text'])

        # X: ランダムにAまたはB（評価者には知らせない）
        x_is_a = random.random() < 0.5
        audio_x = audio_a if x_is_a else audio_b

        test_id = f"abx_{i:03d}_{random.randint(1000, 9999)}"

        save_audio(f"{output_dir}/{test_id}_A.wav", audio_a)
        save_audio(f"{output_dir}/{test_id}_B.wav", audio_b)
        save_audio(f"{output_dir}/{test_id}_X.wav", audio_x)

        tests.append({
            'test_id': test_id,
            'text': sample['text'],
            'x_is_a': x_is_a,  # 正解（評価者には非公開）
            'answer': ''  # 'A' or 'B'
        })

    df = pd.DataFrame(tests)
    # x_is_aは評価者に見せない
    df_for_evaluation = df.drop(columns=['x_is_a'])
    df_for_evaluation.to_csv(f"{output_dir}/abx_evaluation_sheet.csv", index=False)

    return tests
```

## 評価スクリプト

### 統合評価スクリプト

```python
# tools/comprehensive_evaluation.py
import argparse
from pathlib import Path

def comprehensive_evaluation(model_path, test_dataset, output_dir):
    """包括的評価の実行"""
    results = {}

    print("=== 客観評価 ===")
    # 1. WER/CER
    print("Computing WER/CER...")
    results['wer'] = evaluate_wer(model_path, test_dataset)
    results['cer'] = evaluate_cer_from_wer_results(test_dataset)

    # 2. アクセント精度
    print("Computing accent accuracy...")
    results['accent_accuracy'] = evaluate_accent_accuracy(model_path, test_dataset)

    # 3. MCD
    print("Computing MCD...")
    results['mcd'] = compute_average_mcd(model_path, test_dataset)

    # 4. Prosody Score
    print("Computing prosody score...")
    results['prosody_score'] = evaluate_prosody_score(model_path, test_dataset)

    print("\n=== 主観評価用サンプル生成 ===")
    # 5. MOS test samples
    print("Creating MOS test samples...")
    create_mos_test({'test_model': model_path}, test_dataset, f"{output_dir}/mos")

    # 6. CMOS test samples
    print("Creating CMOS test samples...")
    create_cmos_test('baseline_model', {'test_model': model_path}, test_dataset, f"{output_dir}/cmos")

    # 結果保存
    import json
    with open(f"{output_dir}/objective_results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # レポート生成
    generate_report(results, f"{output_dir}/evaluation_report.md")

    return results

def generate_report(results, output_path):
    """評価レポート生成"""
    report = f"""# CosyVoice 日本語評価レポート

## 客観評価結果

| 指標 | スコア | 目標 | 達成度 |
|-----|--------|------|--------|
| WER | {results['wer']:.1%} | <20% | {'✅' if results['wer'] < 0.2 else '❌'} |
| CER | {results['cer']:.1%} | <10% | {'✅' if results['cer'] < 0.1 else '❌'} |
| アクセント精度 | {results['accent_accuracy']:.1%} | >85% | {'✅' if results['accent_accuracy'] > 0.85 else '❌'} |
| MCD | {results['mcd']:.2f} dB | <5.5 dB | {'✅' if results['mcd'] < 5.5 else '❌'} |
| Prosody Score | {results['prosody_score']:.3f} | >0.75 | {'✅' if results['prosody_score'] > 0.75 else '❌'} |

## 分析

### 強み
- [自動生成]

### 改善点
- [自動生成]

### 次のステップ
- [自動生成]
"""

    with open(output_path, 'w') as f:
        f.write(report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    comprehensive_evaluation(args.model_path, args.test_data, args.output_dir)
```

**使用例**:
```bash
python tools/comprehensive_evaluation.py \
    --model_path exp/japanese_finetuning \
    --test_data data/test_ja.json \
    --output_dir evaluation_results
```

## ベースライン設定

### ベースラインモデル

| モデル | 用途 |
|-------|------|
| CosyVoice2-0.5B（オリジナル） | 改善前の基準 |
| Style-Bert-VITS2 | SOTA日本語TTS |
| Ground Truth | 人間音声（上限） |

### ベースライン測定

```bash
# 1. CosyVoice2オリジナル
python tools/comprehensive_evaluation.py \
    --model_path pretrained_models/CosyVoice2-0.5B \
    --test_data data/test_ja.json \
    --output_dir evaluation_results/baseline_cosyvoice2

# 2. Style-Bert-VITS2（比較用）
python tools/comprehensive_evaluation.py \
    --model_path /path/to/style_bert_vits2 \
    --test_data data/test_ja.json \
    --output_dir evaluation_results/baseline_sbv2

# 3. Ground Truth測定
python tools/evaluate_ground_truth.py \
    --test_data data/test_ja.json \
    --output_dir evaluation_results/ground_truth
```

---

**更新履歴**:
- 2025-01-XX: 初版作成

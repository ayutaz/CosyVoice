# CosyVoice3 ONNX Export 実装ドキュメント

**作成日**: 2026-01-13
**目的**: CosyVoice3をUnity Sentisで動作させるためのONNXエクスポートと推論実装

---

## 1. 概要

### 1.1 目標

CosyVoice3の全パイプラインをONNX形式でエクスポートし、PyTorchなしで完全なTTS推論を実現する。

**達成状況**: ✅ 完了
- 全14個のONNXモデルをエクスポート
- STFT/ISTFTをNumPyで実装（PyTorch不要）
- 新規環境でPyTorchなしでの動作を確認済み

### 1.2 パイプライン構成

```
テキスト入力
    ↓
[Tokenizer] テキスト → トークンID
    ↓
[LLM] Qwen2ベース → 音声トークン生成 (自己回帰)
    ↓
[Flow] DiT + Euler Solver → メルスペクトログラム
    ↓
[HiFT] F0予測 + Source生成 + Decoder → 24kHz音声波形
```

### 1.3 達成状況

| コンポーネント | ONNX変換 | 推論動作 | 備考 |
|--------------|---------|---------|------|
| Text Embedding | ✅ | ✅ | Qwen2 embed_tokens |
| LLM Backbone (Initial) | ✅ | ✅ | KVキャッシュ出力対応 |
| LLM Backbone (Decode) | ✅ | ✅ | KVキャッシュ入出力 |
| LLM Decoder | ✅ | ✅ | logits出力 |
| Speech Embedding | ✅ | ✅ | 音声トークン埋め込み |
| Flow Token Embedding | ✅ | ✅ | |
| Flow Pre-Lookahead | ✅ | ✅ | |
| Flow Speaker Projection | ✅ | ✅ | |
| Flow Estimator | ✅ | ✅ | DiT |
| HiFT F0 Predictor | ✅ | ✅ | FP32必須 |
| HiFT Source Generator | ✅ | ✅ | FP32必須 |
| HiFT Decoder | ✅ | ✅ | FP32必須 |

---

## 2. 生成されたONNXファイル一覧

### 2.1 ファイル構成

```
pretrained_models/Fun-CosyVoice3-0.5B/onnx/
├── text_embedding_fp32.onnx           # 544MB - Qwen2テキスト埋め込み
├── llm_speech_embedding_fp16.onnx     # 12MB  - 音声トークン埋め込み
├── llm_speech_embedding_fp32.onnx     # 24MB
├── llm_backbone_initial_fp16.onnx     # 717MB - LLM初回パス
├── llm_backbone_initial_fp32.onnx     # 1.4GB
├── llm_backbone_decode_fp16.onnx      # 717MB - LLMデコードステップ
├── llm_backbone_decode_fp32.onnx      # 1.4GB
├── llm_decoder_fp16.onnx              # 12MB  - logits出力層
├── llm_decoder_fp32.onnx              # 24MB
├── flow_token_embedding_fp16.onnx     # 1MB
├── flow_pre_lookahead_fp16.onnx       # 1MB
├── flow_speaker_projection_fp16.onnx  # 31KB
├── flow.decoder.estimator.fp16.onnx   # 664MB - Flow DiT
├── hift_f0_predictor_fp32.onnx        # 13MB  - FP32必須
├── hift_source_generator_fp32.onnx    # 259MB - FP32必須
├── hift_decoder_fp32.onnx             # 70MB  - FP32必須
├── campplus_fp16.onnx                 # 15MB  - 話者埋め込み
└── speech_tokenizer_v3_fp16.onnx      # 485MB - 音声トークナイザー
```

### 2.2 合計サイズ

- **FP16構成**: 約2.2GB
- **FP32構成**: 約4.4GB

---

## 3. LLM ONNX エクスポート詳細

### 3.1 アーキテクチャ

```
CosyVoice3LM
├── llm: Qwen2Encoder
│   └── model: Qwen2ForCausalLM
│       └── model: Qwen2Model (24層, 896 hidden, 2 KV heads)
├── speech_embedding: nn.Embedding(6761, 896)
└── llm_decoder: nn.Linear(896, 6761)
```

### 3.2 特殊トークン

| トークン | ID | 用途 |
|---------|-----|------|
| SOS | 6561 | シーケンス開始 |
| EOS | 6562 | シーケンス終了 |
| TASK_ID | 6563 | タスク識別 |

Speech Token Size = 6561 (0-6560が音声トークン)

### 3.3 KVキャッシュ形式

```
Shape: [num_layers * 2, batch, num_kv_heads, seq_len, head_dim]
     = [48, 1, 2, seq_len, 64]

# 48 = 24 layers × 2 (key + value)
# 2 = num_key_value_heads (Grouped Query Attention)
# 64 = head_dim
```

### 3.4 エクスポートの重要ポイント

#### Attention実装の切り替え

SDPAはONNXと互換性がないため、`eager`アテンションに切り替える必要がある:

```python
# 元のQwen2Modelの設定を取得
original_config = qwen2_model.config

# eager attentionに変更
new_config = copy.deepcopy(original_config)
new_config._attn_implementation = "eager"

# 新しいモデルを作成して重みをロード
new_model = Qwen2Model(new_config)
new_model.load_state_dict(original_state_dict)
```

#### KVキャッシュの初回出力 (重要な修正点)

**問題**: 初回パスでKVキャッシュを出力しないと、後続のデコードで正しい音声が生成されない。

**修正前**: `use_cache=False` でKVキャッシュなし
**修正後**: `use_cache=True` でKVキャッシュを出力

```python
class LLMBackboneInitialWrapper(nn.Module):
    def forward(self, inputs_embeds, attention_mask):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,  # 重要: KVキャッシュを出力
        )
        hidden_states = outputs.hidden_states[-1]

        # KVキャッシュをフラット化してONNX互換形式に
        past_key_values = outputs.past_key_values
        past_flat = []
        for i in range(self.num_layers):
            key, value = past_key_values[i]
            past_flat.append(key)
            past_flat.append(value)
        past_key_values_flat = torch.stack(past_flat, dim=0)

        return hidden_states, past_key_values_flat
```

### 3.5 LLM推論フロー

```
1. 入力準備
   - SOS埋め込み: speech_embedding(6561) → [1, 1, 896]
   - テキスト埋め込み: text_embedding(tokens) → [1, N, 896]
   - TASK_ID埋め込み: speech_embedding(6563) → [1, 1, 896]
   - 連結: [SOS, text_emb, TASK_ID] → [1, N+2, 896]

2. 初回パス (llm_backbone_initial)
   入力: inputs_embeds [1, N+2, 896], attention_mask [1, N+2]
   出力: hidden_states [1, N+2, 896], kv_cache [48, 1, 2, N+2, 64]

3. 自己回帰ループ
   for each step:
       a. logits = llm_decoder(hidden_states[:, -1:, :])
       b. token = top_k_sample(logits)
       c. if token == EOS: break
       d. next_emb = speech_embedding(token)
       e. hidden, kv_cache = llm_backbone_decode(next_emb, mask, kv_cache)
```

---

## 4. Flow ONNX エクスポート詳細

### 4.1 コンポーネント

| モデル | 入力 | 出力 |
|--------|------|------|
| flow_token_embedding | token [B, T] | embedded [B, T, 512] |
| flow_pre_lookahead | embedded [B, T, 512] | h [B, T, 80] |
| flow_speaker_projection | embedding [B, 192] | spks [B, 80] |
| flow_estimator (DiT) | x, mask, mu, t, spks, cond | velocity [B, 80, T] |

### 4.2 Euler Solver

```python
n_timesteps = 10
x = random_noise([1, 80, mel_len])

for step in range(n_timesteps):
    t = step / n_timesteps
    velocity = flow_estimator(x, mask, mu, t, spks, cond)
    dt = 1.0 / n_timesteps
    x = x + velocity * dt

mel = x
```

### 4.3 注意点

Flow Estimatorはバッチサイズ2を期待するため、推論時に入力を複製する:

```python
x_batch = np.concatenate([x, x], axis=0)  # [2, 80, T]
# 他の入力も同様に複製
```

---

## 5. HiFT ONNX エクスポート詳細

### 5.1 パイプライン

```
mel [1, 80, T]
    ↓
[F0 Predictor] → f0 [1, T]
    ↓
[Source Generator] → source [1, T*256]
    ↓
[STFT] → source_stft [1, 18, T*64]
    ↓
[Decoder] mel + source_stft → magnitude [1, 9, T*64], phase [1, 9, T*64]
    ↓
[ISTFT] → audio [T*256]
```

### 5.2 STFT/ISTFTパラメータ（CosyVoice3固有）

```python
n_fft = 16
hop_len = 4
center = True  # PyTorchデフォルトと同じパディング
audio_limit = 0.99  # 出力クリッピング

# HiFT upsample_rates (CosyVoice3)
upsample_rates = [8, 5, 3]  # 合計120倍
# 注: CosyVoice2は[8, 8]で64倍

# 期待されるSTFTフレーム数
stft_frames = mel_frames * 120 + 1
```

### 5.3 FP32が必要な理由

**問題**: F0 Predictorの中間値が大きすぎてFP16で精度が失われる。

調査結果:
```
FP16 max diff: 1.572403
FP32 max diff: 0.002655
中間値の最大: ~20319 (FP16の表現範囲を超える)
```

**解決策**: HiFT全体をFP32で維持する。

### 5.4 STFT/ISTFT実装（Pure NumPy）

PyTorchフリー環境のため、NumPy/SciPyでSTFT/ISTFTを実装。PyTorchの`center=True`デフォルトと互換性を保つ:

```python
def _stft(self, x: np.ndarray, n_fft: int = 16, hop_len: int = 4, center: bool = True) -> tuple:
    """NumPyベースSTFT（PyTorch互換）"""
    from scipy.signal import get_window

    window = get_window("hann", n_fft, fftbins=True).astype(np.float32)
    x = x.astype(np.float32)

    # PyTorchのcenter=Trueと同じパディング
    if center:
        pad_len = n_fft // 2
        x = np.pad(x, (pad_len, pad_len), mode='reflect')

    n_frames = 1 + (len(x) - n_fft) // hop_len
    n_freqs = n_fft // 2 + 1
    real = np.zeros((n_freqs, n_frames), dtype=np.float32)
    imag = np.zeros((n_freqs, n_frames), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_len
        frame = x[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        real[:, i] = np.real(spectrum)
        imag[:, i] = np.imag(spectrum)

    return real, imag

def _istft(self, real: np.ndarray, imag: np.ndarray, n_fft: int = 16, hop_len: int = 4, center: bool = True) -> np.ndarray:
    """NumPyベースISTFT（PyTorch互換）"""
    from scipy.signal import get_window

    window = get_window("hann", n_fft, fftbins=True).astype(np.float32)
    complex_spec = real + 1j * imag  # [n_freqs, n_frames]
    n_frames = complex_spec.shape[1]

    # 出力長を計算（パディング込み）
    output_length = n_fft + (n_frames - 1) * hop_len
    output = np.zeros(output_length, dtype=np.float32)
    window_sum = np.zeros(output_length, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_len
        frame = np.fft.irfft(complex_spec[:, i], n=n_fft).astype(np.float32)
        output[start:start + n_fft] += frame * window
        window_sum[start:start + n_fft] += window ** 2

    # 窓関数の正規化
    window_sum = np.maximum(window_sum, 1e-8)
    output = output / window_sum

    # center=Trueの場合、パディング除去
    if center:
        pad_len = n_fft // 2
        output = output[pad_len:-pad_len] if len(output) > 2 * pad_len else output

    return output
```

**注意**: この実装はPyTorchを必要とせず、完全にNumPy/SciPyで動作します。

---

## 6. 発見された問題と解決策

### 6.1 KVキャッシュ未初期化問題

**症状**:
- 音声が異常に長い（8.96秒 vs 期待される3秒程度）
- 意味不明な音声

**原因**:
`LLMBackboneInitialWrapper`が`use_cache=False`でエクスポートされており、初回パスでKVキャッシュが生成されなかった。Pure ONNX推論ではKVキャッシュをゼロ初期化していたため、LLMが正しく機能しなかった。

**修正**:
1. `export_llm_onnx.py`で`use_cache=True`に変更
2. ONNXモデルを再エクスポート
3. 推論コードで初回パスからKVキャッシュを取得

### 6.2 HiFT FP16精度問題

**症状**:
- F0予測値がPyTorchと大きく異なる
- 音質の劣化

**原因**:
F0 Predictorの中間層で値が非常に大きくなり（最大20319）、FP16の精度では表現できない。

**解決策**:
HiFT全コンポーネント（F0 Predictor, Source Generator, Decoder）をFP32で維持。

### 6.3 STFT/ISTFT互換性問題（解決済み）

**症状**:
- 音声の振幅が異常（-24〜+24の範囲）
- ノイズが多い
- Griffin-Limフォールバック時に音声がこもる

**原因**:
1. scipy.signal.stftとtorch.stftの正規化が異なる
2. PyTorchのデフォルト`center=True`によるパディングを考慮していなかった
3. CosyVoice3のupsample_rates=[8,5,3]（120倍）を[8,8]（64倍）と誤認

**解決策**:
1. NumPy/SciPyで`center=True`パディング付きSTFT/ISTFTを実装
2. 正しいパラメータを使用: n_fft=16, hop_len=4, center=True
3. 期待フレーム数 = mel_frames × 120 + 1 で検証

**結果**: PyTorchなしで高品質な音声生成を実現。

### 6.4 先頭トークン発音問題（調査完了）

**症状**:
- 音声の先頭に「a〜」という不要な音が入る
- "Hello"が"a tarou"のように聞こえる場合がある

**調査結果**:

| 項目 | ONNX | PyTorch | 差分 |
|------|------|---------|------|
| Hidden states | - | - | max 0.052 |
| Logits | - | - | max 0.026 |
| Top-10トークン順位 | [1,0,244,243,...] | [1,0,244,243,...] | 同一 |
| Greedy生成トークン | [1,0,1,1,1,1,...] | [1,0,1,1,1,1,...] | 同一 |

**結論**: **ONNX特有の問題ではない**

ONNXとPyTorchは完全に同じトークンを生成。差分は数値誤差レベル（max 0.05）。

**根本原因**:
1. プロンプト音声なしでの推論時の期待される動作
2. 話者埋め込みがないため、モデルはデフォルトの音声特性で生成
3. 先頭の低番号トークン（27, 28, 1など）が特定の音に対応

**重要**: CosyVoiceは音声クローニングTTSシステムのため、**プロンプト音声は必須**:
- プロンプト音声から話者埋め込み（campplus）を抽出
- プロンプト音声から音声トークン（speech_tokenizer）を抽出してFlowに渡す
- プロンプト音声のメル特徴量をFlowのコンディショニングに使用

プロンプト音声なしでは:
- ランダムな話者埋め込み → 不自然な声
- Flowコンディショニングなし → 品質低下、先頭の「a~」音

### 6.5 推論スクリプトの使い方

```bash
# プロンプト音声付き推論（必須）
python scripts/onnx_inference_pure.py \
    --text "<|en|>Hello world" \
    --prompt_wav asset/cross_lingual_prompt.wav \
    --output output.wav

# 日本語の場合
python scripts/onnx_inference_pure.py \
    --text "<|ja|>こんにちは" \
    --prompt_wav my_voice.wav \
    --output japanese_output.wav
```

**プロンプト音声の要件**:
- 長さ: 3〜10秒推奨
- フォーマット: WAV（librosa経由で他形式も対応）
- 品質: 明瞭な音声、背景ノイズ最小限

### 6.6 テスト用プロンプト音声

`asset/prompts/` に英語のテスト用プロンプト音声が用意されている：

| ボイス | 性別 | ファイル例 |
|-------|------|-----------|
| Nova | 女性 | `en_female_nova_greeting.wav` |
| Shimmer | 女性 | `en_female_shimmer_greeting.wav` |
| Alloy | 女性 | `en_female_alloy_greeting.wav` |
| Echo | 男性 | `en_male_echo_greeting.wav` |
| Fable | 男性 | `en_male_fable_greeting.wav` |
| Onyx | 男性 | `en_male_onyx_greeting.wav` |

各ボイスに3種類のテキスト（greeting, story, technical）がある。
詳細は `asset/prompts/README.md` を参照。

### 6.7 Flow出力の処理

Flowモデルはプロンプト部分と生成部分を含むメルスペクトログラムを出力する：

```
[prompt_mel (mel_len1)] + [generated_mel (mel_len2)] = total_mel
```

生成音声のみを出力するため、プロンプト部分を除去：

```python
# Flow出力からプロンプト部分を除去
mel = mel[:, :, mel_len1:]  # generated部分のみ抽出
```

---

## 7. 推論パフォーマンス

### 7.1 テスト環境

- CPU: Intel Core (詳細未取得)
- メモリ: 充分
- ONNX Runtime: CPU Provider

### 7.2 結果

テスト文: `"<|en|>Hello world."`

| フェーズ | 処理時間 |
|---------|---------|
| LLM | 〜数秒 |
| Flow | 〜数秒 |
| HiFT | 〜1秒 |

生成音声: 約3.16秒 (79トークン)

---

## 8. Unity Sentis移植ガイド

### 8.1 必要なONNXファイル

最小構成:
- `text_embedding_fp32.onnx`
- `llm_backbone_initial_fp16.onnx`
- `llm_backbone_decode_fp16.onnx`
- `llm_decoder_fp16.onnx`
- `llm_speech_embedding_fp16.onnx`
- `flow_token_embedding_fp16.onnx`
- `flow_pre_lookahead_fp16.onnx`
- `flow_speaker_projection_fp16.onnx`
- `flow.decoder.estimator.fp16.onnx`
- `hift_f0_predictor_fp32.onnx`
- `hift_source_generator_fp32.onnx`
- `hift_decoder_fp32.onnx`

プロンプト音声処理用（必須）:
- `campplus.onnx` - 話者埋め込み抽出（モデルディレクトリ直下）
- `speech_tokenizer_v3.onnx` - 音声トークン抽出（モデルディレクトリ直下）

### 8.2 C#で実装が必要な部分

1. **トークナイザー**: Qwen2 BPEトークナイザーのC#実装
2. **STFT/ISTFT**: n_fft=16, hop_len=4, center=TrueのミニSTFT/ISTFT
   - Python実装（`onnx_inference_pure.py`の`_stft`/`_istft`）を参照
   - 16点FFTで計算量は少ない
3. **Top-K サンプリング**: 確率的トークン選択
4. **Euler Solver**: Flow Matchingの数値積分
5. **KVキャッシュ管理**: テンソル連結と更新
6. **メル特徴量抽出**: プロンプト音声からメルスペクトログラム生成（librosa互換）

### 8.3 注意点

1. **Opset Version**: 15でエクスポート済み（Sentis互換）
2. **動的形状**: 全モデルでdynamic_axes設定済み
3. **バッチサイズ**: Flow Estimatorはバッチ2を期待
4. **数値精度**: HiFTはFP32必須

---

## 9. スクリプト一覧

### 9.1 エクスポートスクリプト

| スクリプト | 用途 |
|-----------|------|
| `scripts/export_llm_onnx.py` | LLMコンポーネントのエクスポート |
| `scripts/export_hift_onnx.py` | HiFTコンポーネントのエクスポート |
| `scripts/export_flow_onnx.py` | Flowコンポーネントのエクスポート |
| `scripts/export_remaining_onnx.py` | その他のコンポーネント |
| `scripts/convert_to_fp16.py` | FP16変換 |

### 9.2 推論スクリプト

| スクリプト | 用途 |
|-----------|------|
| `scripts/onnx_inference_pure.py` | Pure ONNX推論（メイン） |
| `scripts/onnx_inference_hybrid.py` | ハイブリッド推論（参照用） |

### 9.3 テスト・デバッグスクリプト

| スクリプト | 用途 |
|-----------|------|
| `scripts/test_onnx_components.py` | コンポーネント単位テスト |
| `scripts/debug_first_tokens.py` | 先頭トークンデバッグ（調査完了） |

---

## 10. 今後の課題

### 10.1 高優先度

1. **Unity Sentisでの動作検証**
   - 各ONNXモデルのロード確認
   - パフォーマンス測定

2. **C#実装**
   - Qwen2 BPEトークナイザー
   - KVキャッシュ管理
   - Top-Kサンプリング

### 10.2 中優先度

3. **C# ISTFT実装**
   - n_fft=16の小規模ISTFTは実装が容易
   - 16点FFTで計算量も少ない

4. **プロンプト音声対応**
   - 話者埋め込み抽出（CAMPPlus使用）
   - プロンプト音声トークンの追加

### 10.3 低優先度

5. **モデルサイズ最適化**
   - INT8量子化の検討（音質劣化に注意）
   - モデル蒸留

6. **ストリーミング対応**
   - チャンク単位での生成

---

## 11. 参考資料

- [HuggingFace Optimum ONNX Export](https://huggingface.co/docs/optimum-onnx/)
- [Unity Sentis Documentation](https://docs.unity3d.com/Packages/com.unity.ai.inference/)
- [Esperanto: SLM ONNX with KV Cache](https://www.esperanto.ai/blog/exporting-slms-to-onnx-with-kv-cache-support/)
- CosyVoice公式リポジトリ

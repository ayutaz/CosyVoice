[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=CosyVoice2🤠&text2=Next-Gen%20Streaming%20TTS%20💖%20Qwen2&width=800&height=210)](https://github.com/Akshay090/svg-banners)

## 👉🏻 CosyVoice 2.0 👈🏻

**最新バージョン - CosyVoice 2.0 (0.5B)**

[Demos](https://funaudiollm.github.io/cosyvoice2/) | [Paper](https://arxiv.org/abs/2412.10117) | [Modelscope](https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B) | [HuggingFace](https://huggingface.co/spaces/FunAudioLLM/CosyVoice2-0.5B)

## Highlight🔥

**CosyVoice 2.0** は、より正確で、より安定した、より高速で、より優れた音声生成機能を提供します。

### 多言語対応
- **対応言語**: 中国語、英語、日本語、韓国語、中国語方言（広東語、四川語、上海語、天津語、武漢語など）
- **クロスリンガル & コードスイッチング**: ゼロショット音声クローニングによる言語横断・コード切り替えシナリオをサポート

### 超低遅延
- **双方向ストリーミング対応**: CosyVoice 2.0はオフラインとストリーミングのモデリング技術を統合
- **高速First Packet合成**: 高品質な音声出力を維持しながら、150ms以下の遅延を実現

### 高精度
- **発音精度向上**: CosyVoice 1.0と比較して発音エラーを30%〜50%削減
- **ベンチマーク達成**: Seed-TTS評価セットのハードテストセットで最低文字エラー率を達成

### 強力な安定性
- **音色の一貫性**: ゼロショットおよびクロスランゲージ音声合成で確実な音声一貫性を保証
- **クロスランゲージ合成**: バージョン1.0と比較して大幅に改善

### 自然な体験
- **韻律と音質の向上**: 合成音声のアライメントが改善され、MOS評価スコアが5.4から5.53に向上
- **感情と方言の柔軟性**: よりきめ細かい感情制御とアクセント調整をサポート

## 主要機能

### Qwen2ベースのアーキテクチャ
- 事前学習済みQwen2ForCausalLMを活用
- 500Mパラメータ
- Bidirectional Streaming: テキストと音声を5:15の比率で混合

### ストリーミング性能
- First Chunk Latency: 150ms以下
- 因果的Flow Matching（ストリーミング対応）
- チャンク単位の処理とKVキャッシュ

### 最適化オプション
- **vLLM統合**: 4倍高速化
- **TensorRT-LLM**: 2.3倍高速化（Triton使用）
- **JIT/ONNX**: 推論最適化

### 強化学習（GRPO）
- WERベースのReward関数
- 発音精度とMOS評価の向上

## Install

### Clone and install

```sh
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
```

### 環境の構築

```sh
# uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# Python 3.10仮想環境の作成
uv venv --python 3.10 .venv

# 仮想環境の有効化
source .venv/bin/activate  # Linux/macOS
# または
.venv\Scripts\activate  # Windows

# 依存パッケージのインストール
uv pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# Sox互換性問題がある場合
# Ubuntu:
sudo apt-get install sox libsox-dev
# CentOS:
sudo yum install sox sox-devel
```

### モデルのダウンロード

CosyVoice2-0.5Bモデルとttsfrdリソースのダウンロードを強く推奨します。

```python
# SDK経由のモデルダウンロード
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

```sh
# Git経由のモデルダウンロード（git lfsが必要）
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

オプションで、`ttsfrd`リソースを解凍し、`ttsfrd`パッケージをインストールすることで、テキスト正規化のパフォーマンスが向上します。

このステップは必須ではありません。`ttsfrd`パッケージをインストールしない場合、デフォルトでwetextを使用します。

```sh
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

## Basic Usage

CosyVoice2-0.5Bの使用を強く推奨します。

```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
```

### CosyVoice2の基本的な使用法

```python
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# NOTE: https://funaudiollm.github.io/cosyvoice2 の結果を再現する場合は、推論時に text_frontend=False を追加してください
# ゼロショット使用法
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('収到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# ゼロショット話者を保存して後で使用
assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True
for i, j in enumerate(cosyvoice.inference_zero_shot('収到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
cosyvoice.save_spkinfo()

# きめ細かい制御（サポートされる制御は cosyvoice/tokenizer/tokenizer.py#L248 を参照）
for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
    torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct使用法
for i, j in enumerate(cosyvoice.inference_instruct2('収到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# ストリーミング使用法、ジェネレーターを入力として使用できます。これはテキストLLMモデルを入力として使用する場合に便利です
# NOTE: LLMは任意の文の長さを処理できないため、基本的な文の分割ロジックが必要です
def text_generator():
    yield '収到好友从远方寄来的生日礼物，'
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐，'
    yield '笑容如花儿般绽放。'
for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```

### CosyVoice2のvLLM使用法

vLLMを推論に使用する場合は、`vllm==v0.9.0`をインストールしてください。古いvLLMバージョンはCosyVoice2推論をサポートしていません。

`vllm==v0.9.0`には多くの特定の要件があります（例: `torch==2.7.0`）。ハードウェアがvLLMをサポートしていない場合に古い環境が破損しないように、新しい環境を作成できます。

```sh
# vLLM用の別仮想環境作成
uv venv --python 3.10 .venv_vllm
source .venv_vllm/bin/activate  # Linux/macOS

# 標準の依存関係をインストール
uv pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# vLLM関連パッケージをインストール
uv pip install vllm==v0.9.0 transformers==4.51.3 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# vLLM推論の実行
python vllm_example.py
```

### Webデモの起動

CosyVoice2を素早く体験するには、Webデモページを使用できます。

詳細はデモWebサイトを参照してください。

```python
# CosyVoice2-0.5Bを使用
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

### 高度な使用法

上級ユーザー向けに、`examples/libritts/cosyvoice2/run.sh`にトレーニングおよび推論スクリプトを提供しています。

### デプロイメント用のビルド

オプションで、サービスデプロイメントが必要な場合は、以下の手順を実行できます。

#### FastAPI / gRPC デプロイメント

```sh
cd runtime/python
docker build -t cosyvoice:v2.0 .

# gRPC使用法
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v2.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/grpc && python3 server.py --port 50000 --max_conc 4 --model_dir pretrained_models/CosyVoice2-0.5B && sleep infinity"
cd grpc && python3 client.py --port 50000 --mode zero_shot

# FastAPI使用法
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v2.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && python3 server.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B && sleep infinity"
cd fastapi && python3 client.py --port 50000 --mode zero_shot
```

#### Nvidia TensorRT-LLMを使用したデプロイメント

TensorRT-LLMを使用してcosyvoice2のLLMを加速すると、HuggingFace transformers実装と比較して4倍の高速化が得られます。

クイックスタート:

```sh
cd runtime/triton_trtllm
docker compose up -d
```

詳細については、[こちら](https://github.com/FunAudioLLM/CosyVoice/tree/main/runtime/triton_trtllm)を確認してください。

## アーキテクチャ

### コアコンポーネント

CosyVoice2は3つの主要モジュールで構成されています:

1. **LLM (Qwen2LM)**: テキストから離散音声トークンへ
   - Qwen2ForCausalLMベース
   - Bidirectional Streaming (5:15混合)
   - 音声トークン数: 6,561

2. **Flow (CausalMaskedDiffWithXvec)**: 音声トークンからメルスペクトログラムへ
   - 因果的Flow Matching
   - UpsampleConformerEncoder (2倍アップサンプリング)
   - static_chunk_size=25、ストリーミング対応

3. **HiFiGAN (HiFTGenerator)**: メルスペクトログラムから音声波形へ
   - F0予測 + NSF
   - アップサンプリング: [8, 5, 3] → 120倍
   - サンプリングレート: 24000 Hz

### 主要な特徴

- **サンプリングレート**: 24000 Hz
- **フレームレート**: 25 Hz
- **パラメータ数**: 500M
- **MOS評価**: 5.53
- **First Chunk Latency**: 150ms以下

## トレーニング

### データ準備

```sh
cd examples/libritts/cosyvoice2

# Stage -1~3: データダウンロード、前処理、Parquet変換
bash run.sh
```

### モデルトレーニング

```bash
# 3つのモデルを順番にトレーニング: llm → flow → hifigan
export CUDA_VISIBLE_DEVICES="0,1,2,3"
for model in llm flow hifigan; do
    torchrun --nproc_per_node=4 \
      cosyvoice/bin/train.py \
      --train_engine torch_ddp \
      --config conf/cosyvoice2.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --model $model \
      --checkpoint pretrained_models/CosyVoice2-0.5B/$model.pt \
      --model_dir exp/cosyvoice2/$model \
      --tensorboard_dir tensorboard/cosyvoice2/$model
done
```

### GRPOトレーニング（強化学習）

```sh
cd examples/grpo/cosyvoice2

# HuggingFace形式への変換 → GRPO訓練 → CosyVoice形式へ変換
bash run.sh
```

## デプロイメント性能

### パフォーマンス比較（L20 GPU）

| デプロイメント方法 | RTF (batch=8) | First Chunk Latency | 相対速度 |
|-------------------|---------------|---------------------|----------|
| HuggingFace実装 | 0.0947 | - | 1.0x |
| TensorRT-LLM | 0.0418 | 189ms | 2.3x |
| vLLM統合 | さらに高速 | - | 4.0x |

### デプロイメント方法の選択

| 方法 | 用途 | レイテンシ | スループット | 実装難易度 |
|------|------|------------|--------------|------------|
| FastAPI | プロトタイピング、小規模 | 中 | 中 | 低 |
| gRPC | 本番環境、中規模 | 低 | 高 | 中 |
| Triton+TensorRT-LLM | 本番環境、大規模、最高性能 | 最低 | 最高 | 高 |

## 技術仕様

### システム要件

**最小要件:**
- GPU: NVIDIA GPU（6GB VRAM以上）
- RAM: 16GB
- ストレージ: 10GB以上

**推奨要件:**
- GPU: NVIDIA GPU（12GB VRAM以上、A100/L20推奨）
- RAM: 32GB以上
- ストレージ: 50GB以上

**トレーニング要件:**
- GPU: 複数のNVIDIA GPU（各16GB VRAM以上）
- RAM: 64GB以上
- ストレージ: 500GB以上

### ソフトウェア要件

- **Python**: 3.10
- **PyTorch**: 2.3.1（標準）、2.7.0（vLLM使用時）
- **CUDA**: 12.1
- **transformers**: 4.51.3
- **vllm**: v0.9.0（オプション）

## Discussion & Communication

[GitHub Issues](https://github.com/FunAudioLLM/CosyVoice/issues)で直接ディスカッションできます。

公式DingDingチャットグループに参加するには、QRコードをスキャンしてください。

<img src="./asset/dingding.png" width="250px">

## Acknowledge

1. [FunASR](https://github.com/modelscope/FunASR)から多くのコードを借用しました。
2. [FunCodec](https://github.com/modelscope/FunCodec)から多くのコードを借用しました。
3. [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)から多くのコードを借用しました。
4. [AcademiCodec](https://github.com/yangdongchao/AcademiCodec)から多くのコードを借用しました。
5. [WeNet](https://github.com/wenet-e2e/wenet)から多くのコードを借用しました。

## Citations

```bibtex
@article{du2024cosyvoice2,
  title={Cosyvoice 2: Scalable streaming speech synthesis with large language models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and Shi, Xian and Lv, Xiang and Zhao, Tianyu and Gao, Zhifu and Yang, Yexin and Gao, Changfeng and Wang, Hui and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}

@article{du2025cosyvoice3,
  title={CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author={Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal={arXiv preprint arXiv:2505.17589},
  year={2025}
}

@inproceedings{lyu2025build,
  title={Build LLM-Based Zero-Shot Streaming TTS System with Cosyvoice},
  author={Lyu, Xiang and Wang, Yuxuan and Zhao, Tianyu and Wang, Hao and Liu, Huadai and Du, Zhihao},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--2},
  year={2025},
  organization={IEEE}
}
```

## Disclaimer

上記のコンテンツは学術目的のみのために提供されており、技術的な能力を示すことを目的としています。一部の例はインターネットから取得したものです。コンテンツがあなたの権利を侵害している場合は、削除を要求するためにご連絡ください。

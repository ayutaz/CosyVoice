# Matcha-TTS Attribution

このディレクトリには、Matcha-TTSプロジェクトのコードが含まれています。

**元のリポジトリ**: https://github.com/shivammehta25/Matcha-TTS.git
**コミット**: dd9105b34bf2be2230f4aa1e4769fb586a3c824e (v0.0.5)
**ライセンス**: MIT License（LICENSEファイルを参照）
**著者**: Shivam Mehta, Ruibo Tu, Jonas Beskow, Éva Székely, Gustav Eje Henter

## 元の論文
Matcha-TTS: A fast TTS architecture with conditional flow matching
Published at ICASSP 2024

## 統合に関する注記
このコードは元々gitサブモジュールとして含まれていましたが、CosyVoiceユーザーの依存関係管理を簡素化するため、通常のディレクトリに変換されました。

## 将来の更新手順
Matcha-TTSを更新する必要がある場合：

1. Matcha-TTSを別途クローン:
   ```bash
   cd /tmp
   git clone https://github.com/shivammehta25/Matcha-TTS.git
   cd Matcha-TTS
   git checkout <desired-commit>
   ```

2. CosyVoiceリポジトリで更新:
   ```bash
   cd /path/to/CosyVoice
   rm -rf third_party/Matcha-TTS/matcha
   cp -r /tmp/Matcha-TTS/matcha third_party/Matcha-TTS/
   ```

3. このファイルのコミットハッシュを更新

4. 動作確認とコミット

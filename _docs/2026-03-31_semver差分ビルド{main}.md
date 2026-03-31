# 2026-03-31 semver差分ビルド{main}

## 実施内容
- `hypura` の SemVer を `0.1.2 -> 0.1.3` へ更新。
- `hypura-sys` の SemVer を `0.1.2 -> 0.1.3` へ更新。
- 既存ワークツリーのまま差分ビルドを実行して、増分コンパイルで再ビルドを確認。

## 変更ファイル
- `Cargo.toml`
- `hypura-sys/Cargo.toml`

## ビルド確認
- コマンド: `cargo build --release`
- 目的: SemVer 更新後の差分ビルド成立確認

# 実装ログ: gitignore / ワークツリー整理

- **日時**: 2026-04-05 19:27:05 +0900（`py -3` で取得）
- **ワークツリー名**: hypura-main

## 実施内容

1. **git worktree**  
   - `git worktree list` の結果、追加ワークツリーはなく `main` のみ。  
   - `git worktree prune` を実行（スタブ整理）。

2. **.gitignore**  
   - ルートの代替 `CARGO_TARGET_DIR` 用ディレクトリを無視: `/target-pc-*/`  
   - 既存の `/target/` はデフォルト `target/` のみ。`target-pc-build` / `target-pc-cuda12` などは別名のため明示パターンが必要。

3. **クリーンアップ**  
   - `git clean -fdX -- target-pc-build target-pc-cuda12` でビルド成果物を削除。  
   - Windows では CMake / ロックにより **Permission denied / Invalid argument** が一部発生。残る場合は Cargo・IDE・ウイルス対策を止めてから `Remove-Item -Recurse -Force` または `rmdir /s /q` で再試行。

## 検証

- `git check-ignore -v target-pc-build` → `.gitignore` の `/target-pc-*/` にマッチすることを確認。

## 備考

- リポジトリ履歴上、同種の整理は `chore(gitignore): ignore custom Rust target output dirs` でも入っている。重複パターンは入れていない。

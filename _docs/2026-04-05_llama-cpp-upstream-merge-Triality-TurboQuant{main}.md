# llama.cpp 公式 master 取り込み（Triality / TurboQuant 維持）

- **記録時刻 (UTC):** 2026-04-04T21:36:57Z（`py -3` で取得）
- **作業ツリー:** `main`
- **サブモジュール:** `vendor/llama.cpp`
- **ブランチ:** `codex/triality-defaults`（リモート `zapabob/codex/triality-defaults` より 184 コミット先行の状態でマージ完了）

## やったこと

1. `vendor/llama.cpp` で `origin`（`https://github.com/ggerganov/llama.cpp.git`）を `fetch`。
2. マージベース `177c75852…` に対し、独自側 **3 コミット**、公式 `master` 側 **約 183 コミット**の差分を `git merge origin/master` で統合。
3. マージコミット: `a5974fd34` — メッセージ: `merge: upstream ggerganov/llama.cpp master; keep Triality/TurboQuant base`
4. マージ前に作業ツリーに未コミット変更があり `merge` がブロックされたため、いったん `git stash push`。**マージ後**に `stash pop` すると公式更新分と大規模に衝突したため、**クリーンなマージ結果を優先**して `git reset --hard a5974fd34` で復元。捨てたくない WIP が `stash@{0}` に残っている（必要なら手で `stash show -p` を確認）。

## 維持したい独自機能の確認（grep サンプル）

マージ直後のツリーに以下が残存:

- `convert_hf_to_gguf.py`: `add_hypura_turboquant_metadata`、`hypura.turboquant.*` メタデータ、`triality_view` マッピング
- `src/llama-kv-cache.{h,cpp}`: `llama-turboquant.h`、ランタイム設定とログ

（論文通りの SO(8) / Triality 経路は従来コミット列＋上記ファイルに依存。）

## 親リポジトリ

- サブモジュール指し先 `a5974fd34` を **`chore(vendor): sync llama.cpp with upstream master (Triality/TurboQuant retained)`** でコミット済み（例: `be57db8`）。

## fork 反映

- `vendor/llama.cpp` で `git push zapabob codex/triality-defaults` 済み（`dbdd39727..a5974fd34`）。

## リリースビルド確認（続き）

- ルートで `$env:RUSTC_WRAPPER=''; cargo build --release` を実行。**exit code 0**、`Finished release profile ... in 31m 13s`。
- 成果物: `target/release/Hypura.exe`、`target/release/kobold_gguf_gui.exe`（`hypura-sys`＋ vendored llama.cpp / CMake 経路を含むワークスペース全体）。
- ログに `Blocking waiting for file lock on artifact directory` が出た場合は、別プロセスの `cargo` と競合しているので、片方を止めるか待つ。
- PowerShell で `cargo ... | Select-Object -Last N` は**ストリーム終端までバッファ**するため、長時間ビルドの進捗確認には向かない。

## 次の作業（任意）

1. `cargo test --release` やスモーク実行。
2. 不要なら `vendor/llama.cpp` 内 `git stash drop`（マージ前スタッシュ）。
3. 親 `main` を `origin` に `git push`（ローカルだけの場合）。

## なんJメモ（雑）

公式 183 コミットぶっこ抜いても TurboQuant の痕跡 grep で生きてたわ。スタッシュ pop は CI yaml とかで全面戦争になったから即座に hard reset した。WIP 捨てたくねえ奴は stash 覗けや。

## CoT（仮説→検証）

| 仮説 | 検証 | 結果 |
|------|------|------|
| 3 コミットだけなら ort マージでコンフリクト少ない | `merge origin/master` 実行 | コンフリクトなしで完了 |
| マージ後も Hypura メタが消えてないか | `triality` / `turboquant` grep | 主要パスに残存 |
| 親 repo はサブモジュール SHA 更新が必要 | `git diff --staged vendor/llama.cpp` | コミット済み |
| release 全体がリンクまで通るか | `cargo build --release` | 成功（`Hypura.exe` 生成） |

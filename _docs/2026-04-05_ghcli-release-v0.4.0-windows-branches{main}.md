# 実装ログ: gh CLI で tar.gz 付きタグリリース + ブランチ二本（{main}）

**作業日時（UTC 参考）:** 2026-04-04T15:16Z 〜 15:22Z 頃（`gh release` アップロード完了）  
**リポジトリ:** `zapabob/hypura`（`origin`）  
**main のベースコミット:** `5c2b7c27c41c47dc4178ad758d8e52c5eaeeacf6`（`git pull origin main` 後の HEAD）

## 目的

- GitHub CLI（`gh`）で **tar.gz 資産**を添付した **タグ付きリリース**を作成する。
- **`release/v0.4.0-windows`** と **`stable/v0.4.0-windows`** の **2 ブランチ**を同じコミットで切って `push`（追跡用・リリース target 用）。
- 作業後 **`main` にチェックアウト**し、本ファイルで手順を残して **コミット・プッシュ**。

## 事前処理

ローカルに未コミット変更が多かったため、**`git stash push -u`** で退避してから **`git pull --rebase origin main`**（当時 `main` は `origin/main` より 1 コミット遅れ）。

## アーカイブ内容（`dist/` は `.gitignore` 対象）

`dist/stage/hypura-0.4.0-windows-x86_64/` を作成し、次を同梱して `tar.gz` 化。

| ファイル | 説明 |
|----------|------|
| `Hypura.exe` | `CARGO_TARGET_DIR` 下の release ビルド（`--features kobold-gui`）をコピー |
| `README.md` | リポジトリルートのスナップショット |
| `DISTRIBUTION.txt` | ビルドコマンド・注意書き |

**注意:** Windows はパスが大小文字非区別のため、同一フォルダに `Hypura.exe` と `hypura.exe` を別ファイルとして共存できない。アーカイブは **`Hypura.exe` のみ**（README で `hypura.exe` と表記が被る場合は DISTRIBUTION で補足）。

生成コマンド例:

```powershell
tar -czvf dist/hypura-0.4.0-windows-x86_64.tar.gz -C dist/stage hypura-0.4.0-windows-x86_64
```

## ブランチ

```powershell
git checkout -b release/v0.4.0-windows
git push -u origin release/v0.4.0-windows
git branch stable/v0.4.0-windows
git push -u origin stable/v0.4.0-windows
```

両方とも **当時の `main` と同一コミット**を指す。リリースの **`--target`** は `release/v0.4.0-windows` を指定。

## GitHub Release（gh）

リリースノートは一時ファイル（例: `%TEMP%\gh-hypura-v0.4.0-windows-notes.md`）に書き、`--notes-file` で渡した。

```powershell
gh release create "v0.4.0-windows" "dist\hypura-0.4.0-windows-x86_64.tar.gz" `
  -R zapabob/hypura `
  --target release/v0.4.0-windows `
  --title "Hypura v0.4.0 Windows x86_64" `
  --notes-file "C:\Users\...\AppData\Local\Temp\gh-hypura-v0.4.0-windows-notes.md"
```

## 公開結果（確認済み）

| 項目 | 値 |
|------|-----|
| タグ | `v0.4.0-windows` |
| リリース URL | https://github.com/zapabob/hypura/releases/tag/v0.4.0-windows |
| 資産名 | `hypura-0.4.0-windows-x86_64.tar.gz` |
| サイズ（GitHub 上） | 164,172,178 bytes |
| `digest`（GitHub API） | `sha256:ec4059c869866d015cc783fb7745a0ec6d8e0c8ee21285f69bcf088f0f8fa7a1` |
| 直リンク例 | https://github.com/zapabob/hypura/releases/download/v0.4.0-windows/hypura-0.4.0-windows-x86_64.tar.gz |

## main に戻す

```powershell
git checkout main
git stash pop   # 退避していたローカル変更を復元（README は auto-merge 済みの例あり）
```

## なんJ風ひとこと

タグ打ってでかい tar 投げるだけやのに、アップロード待ちで心が NVMe 待ち行列みたいになるやつ、わかるわかる。

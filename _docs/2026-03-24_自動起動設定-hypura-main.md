# 実装ログ: Windows起動時のHypura自動起動

- 対象: `hypura-central-smart.ps1`
- 日時: 2026-03-24

## 実施内容

1. `schtasks` で `HypuraAutoStart` 登録を試行
2. 権限エラー（アクセス拒否）を確認
3. 代替としてユーザー Startup フォルダへ `HypuraAutoStart.cmd` を作成

## 作成ファイル

- `C:\Users\downl\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\HypuraAutoStart.cmd`

内容:

```cmd
@echo off
start "" powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "C:\Users\downl\Desktop\hypura-main\hypura-main\scripts\hypura-central-smart.ps1"
```

## 補足

- これで「Windowsログオン時」に自動起動される。
- 「電源投入直後（ログオン前）」で動かすには、管理者権限でタスクスケジューラの `ONSTART` を構成する必要あり。

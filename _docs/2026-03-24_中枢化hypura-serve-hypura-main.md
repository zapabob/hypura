# 実装ログ: Hypura 中枢化（RTX30 dist + F: GGUF）

- **日時 (UTC)**: 2026-03-24T00:38:06Z 付近（`py -3` で取得）
- **作業ツリー**: `hypura-main`
- **目的**: `dist\hypura-rtx30-windows-stable-2026-03-24\hypura.exe` と  
  `F:\Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf` を **単一のローカル LLM 中枢**（Ollama 互換 API）として固定する。

---

## なんｊ風まとめ

- **結論**: 「中枢」＝ **127.0.0.1:8080 で `hypura serve` 一本**。OpenClaw 等は全部そこを向ければええやんけ。
- **仮説**: Q6_K の 27B は VRAM 食うから、まず `--context 2048` くらいから試して OOM したら下げるのが吉。
- **検証**: `Test-Path` で `hypura.exe` と GGUF は **両方 True** だったわ。ナイス F:。

---

## CoT（ざっくり）

1. **何を中枢にするか** → HTTP で `/api/chat` 等が叩ける **Hypura の serve プロセス**。
2. **バイナリはどれか** → ビルド済み `dist\...\hypura.exe`（ユーザー指定フォルダ）。
3. **モデルはどれか** → ユーザー指定の F: 上 GGUF（パス固定）。
4. **クライアントは** → OpenClaw は `models.providers.ollama.baseUrl` を `http://127.0.0.1:8080` に。

---

## 起動（推奨）

リポジトリ直下から:

```powershell
.\scripts\hypura-central-serve.ps1
```

### 環境変数で上書き

| 変数 | 例 | 意味 |
|------|-----|------|
| `HYPURA_EXE` | `C:\...\hypura.exe` | バイナリ |
| `HYPURA_MODEL` | `F:\....gguf` | GGUF |
| `HYPURA_HOST` | `127.0.0.1` | バインド |
| `HYPURA_PORT` | `8080` | ポート |
| `HYPURA_CONTEXT` | `1024`〜`4096` | コンテキスト長（VRAM に応じて調整） |

---

## OpenClaw 側（中枢へ接続）

```text
openclaw config set models.providers.ollama.baseUrl "http://127.0.0.1:8080"
```

または `~/.openclaw/openclaw.json` の `models.providers.ollama.baseUrl` を同一 URL に。

---

## スモーク（任意）

```powershell
Invoke-WebRequest http://127.0.0.1:8080/
Invoke-WebRequest http://127.0.0.1:8080/api/tags
```

---

## 注意（MILSPEC めいたやつ）

- 27B × Q6_K は **ディスク／RAM／VRAM** を食う。初回は `--context` を控えめに。
- 他プロセスが 8080 を掴んでいたら `HYPURA_PORT` を変えるか解放する。
- 本リポジトリの `dist/` は容量が大きいので **Git にはコミットしない**（`.gitignore` 想定）。

---

## 変更ファイル

- `scripts/hypura-central-serve.ps1` — 中枢起動ラッパー（新規）
- 本ログ — `_docs/2026-03-24_中枢化hypura-serve-hypura-main.md`

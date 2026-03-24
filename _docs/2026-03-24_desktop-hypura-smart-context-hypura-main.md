# 実装ログ: デスクトップショートカット改良 + 2048→8192 + KV 自動コンパクト

- **UTC (py -3)**: `2026-03-24T00:48:36+00:00` 付近（`datetime.now(timezone.utc).isoformat()`）
- **worktree**: `hypura-main`

## 要件

- 既存デスクトップ `.lnk` を **Hypura 中枢ランチャー**（`hypura-central-smart.ps1`）に差し替え。
- **初回・既定は `--context 2048`**。快調なら **`8192`** へ。
- **8192 時の「自動コンパクト」**: Hypura 本体の `KvCacheManager`（`src/cache/kv_cache.rs`）＋プレースメントの hot/warm KV（`placement.rs` の `compute_kv_cache_plan`）。**追加 CLI フラグは不要**（`serve --context 8192` で n_ctx が伸び、長文脈時にウィンドウ外を `llama_memory_seq_rm` で間引き）。

## 追加ファイル

| ファイル | 役割 |
|----------|------|
| `scripts/hypura-central-smart.ps1` | 状態ファイル `%LOCALAPPDATA%\Hypura\central-state.json` に従い `2048`/`8192` で起動。`-PromoteTo8192` / `-ResetTo2048` / `-SmokeAndPromote` / `-ShowState` |
| `scripts/hypura_promote_smoke.py` | `py -3` + **tqdm**（無ければフォールバック）で 2048 一時起動→`/` と `/api/generate` 成功→状態を 8192 に更新 |
| `scripts/Configure-HypuraCentralShortcut.ps1` | デスクトップに合致する `.lnk` を更新し、`Hypura 中枢 (Ollama API).lnk` を常に書き込み |

## 操作

```powershell
# ショートカット作成・既存の更新
.\scripts\Configure-HypuraCentralShortcut.ps1
```

### 昇格（自動スモーク）

- 8080 が空いていること（他プロセスが Hypura を掴んでいないこと）。
- 時間がかかる（モデルロード待ち）。

```powershell
.\scripts\hypura-central-smart.ps1 -SmokeAndPromote
```

### 昇格（手動）

2048 で快調と確認したあと:

```powershell
.\scripts\hypura-central-smart.ps1 -PromoteTo8192
```

### 8192 → 2048 に戻す

```powershell
.\scripts\hypura-central-smart.ps1 -ResetTo2048
```

## 注意（PowerShell 5.1）

- ランチャー内の **実行時メッセージは ASCII 中心**（UTF-8 BOM なし環境での `→` 等の誤パース回避）。コメントは日本語のまま。

## 既知の制約

- `-SmokeAndPromote` は **モデルロード込み**で数分かかることがある（`--max-wait` 既定 180 回 × `--delay` 秒）。
- 既に `http://127.0.0.1:8080` が応答する場合は **拒否**（ポート競合防止）。

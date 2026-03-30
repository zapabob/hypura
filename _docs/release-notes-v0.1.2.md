# Hypura v0.1.2 — stable for RTX 3060 / RTX 3080 (Windows)

## 日本語

### このリリースについて

`v0.1.2` は **Windows + NVIDIA RTX 3060 / 3080（Ampere, `sm_86`）** を主ターゲットにした **安定運用向け** のパッチリリースです。  
OpenClaw（Ollama 互換クライアント）および EasyNovelAssistant（Kobold 互換 HTTP）からの接続を想定し、**API 互換・GUI/CLI 同等性・可観測性** を強化しています。

### 主な変更（ハイライト）

- **OpenClaw / Ollama 互換**
  - `POST /api/show` で `name` および `model` を受理（クライアント差の吸収）。
  - `POST /api/chat` で `think` フィールドを受理（現状は互換のため無害化）。
  - `options` に `num_ctx` を追加（将来のコンテキスト指定拡張に備えた互換受理）。
- **Kobold 互換（EasyNovelAssistant / Koboldcpp 系）**
  - `/api/v1/model`, `/api/v1/generate`, `/api/extra/generate/stream`, `/api/extra/generate/check`, `/api/extra/abort`, `/api/extra/true_max_context_length` を継続提供。
  - Kobold 系リクエストに **TurboQuant ランタイム上書き**（`tq_*`）を渡せるように整理。
  - ストリーム終端に **`tok_per_sec_avg` / `prompt_eval_ms`** を付与し、運用時の目視確認をしやすくしました。
- **Kobold-lite GUI（`/kobold-lite`）**
  - プリセット保存（v2）、プリセット差分、CLI ブリッジ（Export/Import）、Retry Last、ランタイムメトリクス表示などを追加。
- **ビルド / リンク修正**
  - GGML の API 変更に合わせ、`hypura_kv_codec` 内の `ggml_map_custom` → **`ggml_map_custom1`** に置換（Windows リリースリンクエラー解消）。

### 付属アーティファクト

| ファイル | 内容 |
| --- | --- |
| `hypura-rtx3060-3080-stable-v0.1.2.tar.gz` | Windows 向けにビルドした `hypura.exe` と README を同梱した配布用アーカイブ |

展開例（PowerShell）:

```powershell
tar -xzf hypura-rtx3060-3080-stable-v0.1.2.tar.gz
```

### 安定版ブランチ

- ブランチ: **`stable/v0.1.2`**（タグ `v0.1.2` と同じコミットを指します）
- 用途: 本番・長期運用で **固定ビルド** を追いたい場合の追跡用。

### アップグレード時の注意

- 既存の `hypura serve` を上書きする前に、使用中のポート（既定 `8080`）とモデルパスを確認してください。
- 未検証モデルは従来どおり **短い `max_tokens` から** 試してください。

---

## English

### About this release

`v0.1.2` is a **stability-focused patch release** aimed primarily at **Windows + NVIDIA RTX 3060 / RTX 3080 (Ampere, `sm_86`)** deployments.  
It improves compatibility with **OpenClaw (Ollama-compatible clients)** and **EasyNovelAssistant (Kobold-compatible HTTP)**, and tightens **GUI/CLI parity** and **runtime observability**.

### Highlights

- **OpenClaw / Ollama compatibility**
  - `POST /api/show` accepts both `name` and `model`.
  - `POST /api/chat` accepts `think` (accepted for compatibility; currently a no-op).
  - `options` accepts `num_ctx` (forward-compatible hook for context sizing).
- **Kobold compatibility**
  - Continues to expose Kobold-style endpoints including streaming/check/abort and `true_max_context_length`.
  - Kobold requests can pass **TurboQuant runtime overrides** via `tq_*` fields.
  - Stream final NDJSON objects may include **`tok_per_sec_avg`** and **`prompt_eval_ms`** for easier ops debugging.
- **Kobold-lite GUI (`/kobold-lite`)**
  - Presets v2, preset diff, CLI bridge import/export, retry-last, and on-screen runtime metrics.
- **Build fix**
  - Updates KV codec glue to use **`ggml_map_custom1`** (fixes Windows link failures against current vendored GGML).

### Artifact

| Asset | Description |
| --- | --- |
| `hypura-rtx3060-3080-stable-v0.1.2.tar.gz` | Packaged Windows `hypura.exe` plus README for distribution |

### Stable branch

- Branch: **`stable/v0.1.2`** (tracks the same commit as tag `v0.1.2`)
- Use case: pin a **fixed baseline** for long-running production setups.

### Upgrade notes

- Verify your listen host/port (default `8080`) and GGUF path before replacing a running binary.
- For untested models, start with a **small generation budget** first.

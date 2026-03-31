```
 _   _
| | | |_   _ _ __  _   _ _ __ __ _
| |_| | | | | '_ \| | | | '__/ _` |
|  _  | |_| | |_) | |_| | | | (_| |
|_| |_|\__, | .__/ \__,_|_|  \__,_|
       |___/|_|
   メモリに収まらないモデルを動かす / Run models too big for your memory
```

---

## 概要

Hypura はストレージ階層を意識した LLM 推論スケジューラです。
モデルのテンソルを GPU・RAM・NVMe にアクセスパターン・帯域幅・ハードウェア特性に基づいて自動配置し、**物理メモリを超える大規模モデルをクラッシュなしに動作させます**。

- 31 GB の Mixtral 8x7B を 32 GB マシンで **2.2 tok/s** で実行
- 40 GB の Llama 3.3 70B を 32 GB マシンで **0.3 tok/s** で実行
- vanilla llama.cpp は両方 OOM でクラッシュ

**対応プラットフォーム:**

| プラットフォーム | GPU | NVMe I/O |
|---|---|---|
| macOS (Apple Silicon) | Metal (`F_NOCACHE` + `pread`) | ✅ |
| Windows ネイティブ | CUDA RTX 2060+ (`FILE_FLAG_NO_BUFFERING` + `ReadFile`) | ✅ |
| WSL2 (Windows) | CUDA RTX 2060+ (`posix_fadvise` + `pread`) | ✅ |
| Linux | CUDA RTX 2060+ (`posix_fadvise` + `pread`) | ✅ |

---

## なぜ必要か

コンシューマ向けハードウェア（MacBook Pro、RTX 3060 搭載 PC など）は高速な統合メモリや NVMe ストレージを搭載していますが、容量に限界があります。32 GB のマシンで 40 GB のモデルをナイーブに読み込もうとすると、OS がスワップを繰り返し OOM キラーが介入します。

Hypura はモデルアーキテクチャを理解することでこの問題を解決します:

- **Norms・Embeddings** — 小さいが毎トークンアクセスされる → GPU に固定
- **MoE エキスパートルーティング** — スパース性を利用。8 エキスパート中 2 つしか発火しない。ルーターインターセプションで選択されたエキスパートを識別し、GGUF ファイルから必要なストライドのみ NVMe ストリーミング（I/O 75% 削減）。ニューロンキャッシュが時間局所性を活かし 99.5% ヒット率を達成。共活性化追跡で投機的プリフェッチを実現
- **Dense FFN ウェイト** — gate/up/down ウェイト（モデルサイズの約 60%）をプールバッファ経由で NVMe からストリーミング。アテンション・Norms は GPU 常駐

---

## 動作原理

Hypura は GGUF ファイルを読み込み、ハードウェアをプロファイリング（GPU 作業セット、RAM、NVMe 帯域幅）し、すべてのテンソルを最適な階層に割り当てる配置最適化を解きます。

**推論モードの自動選択:**

| モード | 条件 | 説明 |
|---|---|---|
| **Full-resident** | モデルが GPU+RAM に収まる | NVMe I/O なし。フル GPU 速度 |
| **Expert-streaming** | MoE モデル（Mixtral 等）| 非エキスパートテンソル（~1 GB）のみ GPU。エキスパートは NVMe から on-demand ストリーミング |
| **Dense-FFN-streaming** | 大規模 Dense モデル（Llama 70B 等）| アテンション+Norms を GPU に（~8 GB）。FFN テンソルは NVMe からストリーミング |

プールバッファサイズ・プリフェッチ深度・メモリバジェットはハードウェアプロファイルから自動計算されます。

---

## NVMe ストリーミング — Windows 対応

Windows でも macOS と同等の NVMe キャッシュバイパス読み出しが動作します。

| 機能 | macOS | Linux/WSL2 | Windows |
|---|---|---|---|
| キャッシュバイパスオープン | `F_NOCACHE` | `O_DIRECT` | `FILE_FLAG_NO_BUFFERING` |
| ランダムオフセット読み出し | `pread(2)` | `pread(2)` | `ReadFile` + `OVERLAPPED` |
| 匿名ページ割り当て | `mmap(MAP_ANON)` | `mmap(MAP_ANON)` | `VirtualAlloc` |
| ページ解放ヒント | `madvise(MADV_FREE)` | `madvise(MADV_DONTNEED)` | `VirtualFree(MEM_DECOMMIT)` |

これらの違いは `src/io/compat.rs` の統一 API の背後に隠蔽されており、上位レイヤー（IoPool、NvmePrefetcher、iobench）はプラットフォームを意識しません。

---

## パフォーマンス

**M1 Max、32 GB 統合メモリ、NVMe シーケンシャル読み取り ~5.1 GB/s での計測値**

| モデル | サイズ | GPU | NVMe | モード | Hypura | llama.cpp | 備考 |
|---|---|---|---|---|---|---|---|
| Qwen 2.5 14B Q4_K_M | 8.4 GB | 8.4 GB | — | full-resident | **21 tok/s** | ~21 tok/s | GPU 収容、オーバーヘッドなし |
| Mixtral 8x7B Q5_K_M | 30.9 GB | 1.1 GB | 29.8 GB | expert-streaming | **2.2 tok/s** | **OOM** | 全層 Metal、キャッシュヒット率 99.5% |
| Llama 3.3 70B Q4_K_M | 39.6 GB | 7.8 GB | 31.8 GB | dense-FFN-streaming | **0.3 tok/s** | **OOM** | 全層 Metal、24 スロット動的プール、7 層プリフェッチ |

---

## 実運用結果（Windows + OpenClaw） / Real-world Operations (Windows + OpenClaw)

### 日本語

以下は Windows 11 + RTX 30 系環境で、Hypura を OpenClaw の中枢推論サーバーとして運用した際の実測ベースの運用知見です。

- **接続方式**: Ollama 互換 API（`http://127.0.0.1:8080`、`/api/tags` / `/api/generate` / `/api/chat`）
- **運用モデル**: `Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`
- **OpenClaw 折衷設定**: `contextWindow=32768`（警告回避） + `maxTokens=1024`（安定性重視）
- **実運用で効いたポイント**:
  - `low context window` 警告は `contextWindow` 表示値で解消可能
  - OOM（`KV cache full` / `failed to allocate graph`）は `maxTokens` の抑制で大幅緩和
  - モデル名不一致（`primary` vs `/api/tags`）を揃えると接続トラブルが減少
- **自動起動/ヘッドレス運用**: Windows Startup から `hypura-central-smart.ps1` を hidden 起動し、再ログオン後も同設定で復帰

運用上は「見た目のコンテキスト値」と「実際の生成長」を分離して管理するのが有効です。  
長文連続生成が必要なワークロードでは `maxTokens=512` への追加引き下げを推奨します。

### English

These are field-tested operational findings from running Hypura as a central inference server for OpenClaw on Windows 11 + RTX 30-series hardware.

- **Connection**: Ollama-compatible API (`http://127.0.0.1:8080`, using `/api/tags`, `/api/generate`, `/api/chat`)
- **Production model**: `Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`
- **Balanced OpenClaw profile**: `contextWindow=32768` (suppresses warnings) + `maxTokens=1024` (stability-first)
- **What worked in practice**:
  - `low context window` warnings are avoided by increasing the configured `contextWindow`
  - OOM symptoms (`KV cache full`, `failed to allocate graph`) are significantly reduced by capping `maxTokens`
  - Aligning model IDs between `primary` and `/api/tags` prevents common provider mismatch failures
- **Headless autostart**: launch `hypura-central-smart.ps1` hidden from Windows Startup for consistent reboot/logon recovery

In production, separating the "displayed context capacity" from "actual generation budget" is a practical strategy.  
For long continuous generation workloads, lowering `maxTokens` further to `512` is recommended.

---

## インストール

Rust 1.75+ と CMake が必要です（vendored llama.cpp のビルドに使用）。

### macOS (Apple Silicon)

```sh
git clone --recurse-submodules https://github.com/zapabob/hypura.git
cd hypura
cargo build --release
```

### WSL2 (Windows)

CUDA ツールキット（12.x 推奨）がインストールされていることを確認してください。

```sh
# WSL2 ターミナル内で実行
git clone --recurse-submodules https://github.com/zapabob/hypura.git
cd hypura
cargo build --release
```

RTX 3060 以上では CUDA アーキテクチャ `sm_86` がデフォルトターゲットです（20xx: sm_75、40xx: sm_89、H100: sm_90）。

### Windows ネイティブ

```powershell
# PowerShell または Git Bash
git clone --recurse-submodules https://github.com/zapabob/hypura.git
cd hypura

# CUDA_PATH 環境変数を設定（例: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4）
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

cargo build --release
```

バイナリは `target/release/hypura.exe` に生成されます。

---

## クイックスタート

```sh
# ハードウェアプロファイリング（初回のみ、結果はキャッシュされる）
hypura profile

# GGUF モデルで推論
hypura run ./model.gguf --prompt "Hello, world"

# インタラクティブチャット
hypura run ./model.gguf --interactive

# ベンチマーク: Hypura vs ナイーブベースライン
hypura bench ./model.gguf

# モデル配置プランの確認（ロードなし）
hypura inspect ./model.gguf

# NVMe I/O マイクロベンチマーク
hypura iobench ./model.gguf --read-gb 1.0

# エキスパートレイアウト最適化（Mixtral 等の MoE モデル）
hypura optimize ./model.gguf
```

未テストモデルでは最初に `--max-tokens 10` から始めることを推奨します。

---

## Ollama 互換サーバー

Hypura は Ollama 互換 HTTP API を公開しており、Ollama に対応したツール（[OpenClaw](https://github.com/openclaw/openclaw) など）のドロップイン代替として機能します。

```sh
hypura serve ./model.gguf
# Hypura serving Mixtral 8x7B Instruct v0.1
#   Endpoint: http://127.0.0.1:8080
#   Ollama-compatible API: /api/generate, /api/chat, /api/tags
```

### エンドポイント

| エンドポイント | 説明 |
|---|---|
| `GET /` | ヘルスチェック |
| `GET /api/tags` | 読み込み済みモデル一覧 |
| `GET /api/version` | サーバーバージョン |
| `POST /api/show` | モデルメタデータ |
| `POST /api/generate` | テキスト補完（ストリーミング NDJSON または単一レスポンス）|
| `POST /api/chat` | チャット補完（ストリーミング NDJSON または単一レスポンス）|

### OpenClaw との連携

```sh
openclaw config set models.providers.ollama.baseUrl "http://127.0.0.1:8080"
```

### Kobold-lite / EasyNovelAssistant 運用プリセット（段階昇格）

以下は `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` を使った
Windows + RTX 3060 の安定化手順です。

| 段階 | context | max_tokens | 用途 | 昇格条件 |
|---|---:|---:|---|---|
| **短文** | 4096 | 64 (必要なら 128) | 疎通確認、短い往復 | 3連続成功 |
| **中文** | 4096 | 256 | 通常の執筆補助 | 3連続成功 |
| **長文** | 8192 | 512 | 長めの連続生成 | 中文段階が安定 |

推奨運用:

- `serve` の常用は `--context 4096` を基準にする
- 生成長だけを `64 -> 256 -> 512` へ段階的に上げる
- どこかで不安定化したら直前の成功値へロールバックする

### 実測（latest session, Windows 11 + RTX 3060, v0.1.3）

- `hypura run --context 4096 --max-tokens 32`:
  - Prompt eval: `6145.7 ms (1.1 tok/s)`
  - Generation: `2.5 tok/s (avg)`, generated tokens: `14`
- `hypura run --context 8192 --max-tokens 512`:
  - モデルロードと生成開始を確認（長文では処理時間が増加）
- `hypura serve --context 4096` + proxy (`:5001`) の3連続疎通:
  - `iter=1 gen_chars=89`
  - `iter=2 gen_chars=35`
  - `iter=3 gen_chars=56`
  - `/api/tags` と `/api/v1/model` は全試行で成功

### Kobold-lite / EasyNovelAssistant Staged Presets (safe 4096 -> 8192)

These staged values are validated with
`Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`
on Windows + RTX 3060:

| Stage | context | max_tokens | Primary use | Promotion rule |
|---|---:|---:|---|---|
| **Short** | 4096 | 64 (or 128) | Health check and short replies | 3 consecutive passes |
| **Medium** | 4096 | 256 | Daily drafting | 3 consecutive passes |
| **Long** | 8192 | 512 | Longer generation | Promote only after Medium stability |

Operational baseline:

- Keep `serve --context 4096` as the default baseline
- Increase generation budget incrementally: `64 -> 256 -> 512`
- Roll back to the previous successful stage immediately on instability

---

## アーキテクチャ

Hypura は2つのクレートからなる Cargo ワークスペースです。

- **`hypura`** — メインバイナリ＋ライブラリ。CLI は `src/main.rs`、ロジックは `src/lib.rs` モジュール群
- **`hypura-sys`** — llama.cpp の FFI バインディング（`vendor/llama.cpp/` に vendored、CMake でビルド）

### 主要モジュール

| モジュール | 目的 |
|---|---|
| `io/compat.rs` | プラットフォーム抽象化（macOS/Linux/Windows の I/O プリミティブ統一）|
| `scheduler/placement.rs` | LP + greedy テンソル配置最適化（GPU/RAM/NVMe 階層）|
| `compute/inference.rs` | 推論エンジン: `generate_blocking`、`generate_with_nvme_scheduling` |
| `compute/nvme_backend.rs` | カスタム GGML バッファ型、エキスパート/FFN ストリーミング、ニューロンキャッシュ |
| `server/routes.rs` | Ollama 互換 API の Axum HTTP ハンドラ |
| `profiler/` | ハードウェア検出（CPU/GPU/メモリ帯域幅/NVMe スループット）|
| `cli/bench.rs` | A/B ベンチマークハーネス |
| `model/tensor_role.rs` | テンソル分類（配置スコアリング用）|
| `cache/coactivation.rs` | エキスパート共活性化追跡（投機的プリフェッチ用）|
| `cache/neuron_cache.rs` | 読み込み済みエキスパートスライスの LRU キャッシュ |

---

## FAQ

### SSD が壊れませんか?

いいえ。**Hypura は推論中に SSD への書き込みを一切行いません。**

SSD の劣化は書き込みサイクル（NAND フラッシュの P/E サイクル）によって生じます。読み取りはフラッシュセルを劣化させません。Hypura の NVMe I/O パスは読み取り専用で、GGUF ファイルからテンソルウェイトを RAM/GPU メモリプールにストリーミングするだけです。SSD はコールドストレージとして使用されます。

唯一の書き込みは無視できる程度です: ベンチマーク結果 JSON（~KB）、共活性化統計（~KB）、`hypura optimize` コマンド（任意実行時の1回限り）。通常の推論では SSD 書き込みはゼロです。

### Windows でも NVMe ストリーミングは動きますか?

はい。`FILE_FLAG_NO_BUFFERING` + `ReadFile(OVERLAPPED)` が macOS の `F_NOCACHE` + `pread` と同等の機能を提供します。詳細は `_docs/windows-wsl2-port.md` を参照してください。

---

## 安全上の注意

- `bench --baseline` はモデルが RAM − 4 GB ヘッドルームを超える場合にブロックされます。`--force` で上書き可能ですが自己責任で
- 未テストモデルでは必ず `--max-tokens 10` から始めてください
- テストモデルは `./test-models/` に置いてください（リポジトリには含めない）

---

## ライセンス

MIT

---

## Ethics

このリポジトリのコードは私が自分で書いたものではありません。このプロジェクトは LLM を使って私の指示に基づいてタスクを実行するという探求です。NVMe を活用した推論はメモリの一形態として（低速ではあるが）十分に有効であるにもかかわらず、未活用であるという直感から始まりました。

---
## Release 0.2.0 (Theme-linked Kobold Parity) / リリース 0.2.0（テーマ連動Kobold同等化）

### 日本語

- SemVer 更新: `hypura` を `0.2.0` へ更新（`hypura-sys` は `0.1.3` 維持）。
- 互換継続: OpenClaw / EasyNovelAssistant 向け API 契約（`/api/show` `name` 受理、Kobold 互換導線）を維持。
- GUI 強化: `--ui-theme` と GUI テーマ切替（light/dark/classic）を完全連動。
- Kobold風UI: モード切替・モデル選択・履歴/イベント監視をテーマ別に視認性最適化。
- 運用改善: 生成中モデル切替抑止、サーバー側プリセット/履歴/イベントAPIで運用追跡を強化。

### English

- SemVer bump: `hypura` updated to `0.2.0` (`hypura-sys` remains `0.1.3`).
- Compatibility continuity: OpenClaw / EasyNovelAssistant contracts preserved (`/api/show` with `name`, Kobold-compatible routes).
- GUI uplift: full theme linkage between CLI `--ui-theme` and GUI theme switcher (light/dark/classic).
- Kobold-style UX: themed dashboard cards for mode switching, model switching, history, and event logs.
- Operational safety: model switching remains blocked while generation is active.

---

## Kobold-lite Parity++ (v0.2.0)

### 日本語

- UIモード導線: `Chat` / `Instruct` / `Storywriter` / `Adventure` をワンクリック切替。
- モデル運用: `/api/extra/models` + `/api/extra/model/switch` でGGUF一覧とライブ切替。
- プリセット運用: ブラウザ保存に加えてサーバー側プリセットAPI（`/api/extra/presets/*`）を追加。
- 可観測性: 生成履歴（`/api/extra/history`）とイベントログ（`/api/extra/events`）をGUIへ常時表示。
- CLI追従: `serve` に `--model-dir` と `--ui-theme` を追加し、GUI運用と起動設定の整合を強化。

### English

- UI mode workflow: one-click switching for `Chat` / `Instruct` / `Storywriter` / `Adventure`.
- Model operations: GGUF listing and live switching via `/api/extra/models` and `/api/extra/model/switch`.
- Preset operations: server-side preset APIs (`/api/extra/presets/*`) in addition to browser-local presets.
- Observability: generation history (`/api/extra/history`) and event logs (`/api/extra/events`) rendered directly in GUI.
- CLI follow-up: `serve` supports `--model-dir` and `--ui-theme` for GUI/runtime alignment.
- Theme linkage: GUI theme selector syncs with server theme APIs (`GET/POST /api/extra/ui-theme`) and CLI startup defaults.

---

## English Overview (Short)

Hypura is a storage-tier-aware LLM inference scheduler that places tensors across GPU/RAM/NVMe and runs models that exceed physical memory limits.  
For full technical details, see the bilingual sections above (`概要`, `動作原理`, `アーキテクチャ`, `FAQ`).

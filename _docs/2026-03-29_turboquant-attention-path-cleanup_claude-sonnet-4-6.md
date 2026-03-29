# TurboQuant attention path cleanup

**日付:** 2026-03-29
**モデル:** claude-sonnet-4-6
**対象ブランチ:** main

---

## 背景

attention path completion（Qcur/Kcur/Vcur/kq/kq_soft_max/kqv 対応）実装後のレビューで
特定された課題を修正した。

---

## 変更サマリー

### `src/compute/inference.rs`

| 区分 | 内容 |
|------|------|
| コメント追加 | `observe_eval_callback` 冒頭に loose (`contains`) vs strict (`==`) 照合戦略の説明を追加 |
| メモリ最適化 | Score パス: `query_vectors.get()` → `remove()` で使用後即解放 |
| メモリ最適化 | ValueOutput パス: `softmax_weights.get()` → `remove()` で使用後即解放 |
| 型修正 | `codec.score_k(..., query, ...)` → `codec.score_k(..., &query, ...)` (`Vec<f32>` → `&[f32]` 参照) |
| リネーム | `kq_softmax_callback_hits` → `kq_soft_max_callback_hits` (llama.cpp テンソル名に統一) |
| リネーム | `kq_softmax_hits` → `kq_soft_max_hits` (RuntimeCallbackStats + ログ出力 3 箇所) |
| テスト追加 | `runtime_tensor_component_maps_attention_nodes` — 6 コンポーネント型マッピング |
| テスト追加 | `tensor_vector_range_accounts_for_streams` — ストリーム分割インデックス |

**メモリ改善:** `query_vectors` / `softmax_weights` のエントリを使用後すぐに削除することで
メモリが `O(tokens × layers × heads)` から `O(batch × layers × heads)` になった。

### `hypura-sys/build.rs`

| 区分 | 内容 |
|------|------|
| bindgen 設定 | `.layout_tests(false)` 追加 — Windows で `llama_sampler` がオペーク型として生成される際の誤ったサイズアサーション (`E0080`) を抑制 |
| cmake 設定 | `CMAKE_BUILD_TYPE=Release`, `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL` 追加 |
| C シム | `hypura_kv_codec.c` / `.h` をビルドに追加 |

### `Cargo.toml`

| 区分 | 内容 |
|------|------|
| windows-sys フィーチャー | `Win32_Security` を追加 — `CreateFileW` の `SECURITY_ATTRIBUTES` 型依存を解決 |

### `src/io/compat.rs`

| 区分 | 内容 |
|------|------|
| Windows API 互換 | `CreateFileW` 戻り値 `HANDLE (*mut c_void)` を `as isize` でキャスト |
| Windows API 互換 | `INVALID_HANDLE_VALUE` 比較を `handle as isize == INVALID_HANDLE_VALUE` に変更 |
| Windows API 互換 | `CreateFileW` の `hTemplatefile` 引数を `0` → `std::ptr::null_mut()` に変更 |
| Windows API 互換 | `CloseHandle(fd)` → `CloseHandle(fd as *mut _)` にキャスト |
| Windows API 互換 | `ReadFile(fd, ...)` → `ReadFile(fd as *mut _, ...)` にキャスト |

背景: `windows-sys v0.59` で `HANDLE` 型が `isize` から `*mut core::ffi::c_void` に変更された。

### `src/profiler/cpu.rs`

| 区分 | 内容 |
|------|------|
| API 互換 | `sys.physical_cpu_count()` → `sys.physical_core_count()` (sysinfo v0.31 API 変更) |

---

## TurboQuant モード設計確認

レビュー中に確認した設計意図:

| モード | kq 上書き | kqv 上書き | 根拠 |
|--------|-----------|------------|------|
| `Exact` | なし | なし | callback 登録なし |
| `PaperKeyOnly` | あり (score_k) | なし | K のみ再構築 |
| `PaperFullKv` | あり (score_k) | あり (read_v) | K + V 両方再構築、論文通り |

`PaperFullKv` で kq も上書きするのは、両モードとも K を再構築するため正しい。

---

## Windows ビルド制限事項

- `cargo check` / `cargo test` は Avast / Windows Defender リアルタイム保護が
  コンパイル済みオブジェクトを隔離するため失敗することがある
- 管理者 PowerShell で AV 除外を追加してからビルドすること:
  ```powershell
  Add-MpPreference -ExclusionPath "$env:USERPROFILE\.cargo"
  Add-MpPreference -ExclusionPath "C:\Users\downl\Desktop\hypura-main\hypura-main\target"
  ```
- `llama_sampler` サイズアサーション: `.layout_tests(false)` で抑制済み
  (ポインタ経由でのみ使用するため実害なし)

---

## 検証コマンド

```powershell
# 管理者 PowerShell
cd C:\Users\downl\Desktop\hypura-main\hypura-main
$env:CARGO_INCREMENTAL = "0"
cargo fmt
cargo test --lib -- compute::inference
# 期待: runtime_tensor_component_maps_attention_nodes ok
#       tensor_vector_range_accounts_for_streams ok
```

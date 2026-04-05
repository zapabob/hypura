# 実装ログ: ストレージ用語整理とマルチプール予約（{main}）

**日付:** 2026-04-04（作業環境時計）  
**worktree:** main 相当チェックアウト

## 目的

プラン「重み削減と SSD 分散で大規模モデルを動かせるか」の実装タスク:

1. README に **NVMe は SSD の一種**、**複数物理ドライブへのテンソル分散は未実装**を明記。
2. コード側に **単一チャネル帯域モデルの集約関数**と **`PrefetchOp::storage_pool_id` 予約**、`IoPool` のドキュメントを追加。

## 変更サマリ

| 領域 | 内容 |
|------|------|
| [README.md](../README.md) | 日英で用語・制限・`storage_pool_id` 予約を追記 |
| [hypura/src/scheduler/types.rs](../hypura/src/scheduler/types.rs) | `SecondaryStoragePoolId`, `DEFAULT_SECONDARY_STORAGE_POOL`, `StorageTier::Nvme` / `PrefetchOp` 拡張 |
| [hypura/src/profiler/types.rs](../hypura/src/profiler/types.rs) | `effective_secondary_storage_peak_bw()` — プロファイル済み NVMe/SATA の **最大** `peak_sequential`（単一チャネル前提の代表帯域） |
| [hypura/src/scheduler/placement.rs](../hypura/src/scheduler/placement.rs) | `compute_tier_capacities` / `quick_estimate` で上記関数を使用 |
| [hypura/src/scheduler/prefetch.rs](../hypura/src/scheduler/prefetch.rs) | 同上 + `PrefetchOp` 生成時に `DEFAULT_SECONDARY_STORAGE_POOL` |
| [hypura/src/scheduler/estimator.rs](../hypura/src/scheduler/estimator.rs) | 同上 |
| [hypura/src/compute/nvme_backend.rs](../hypura/src/compute/nvme_backend.rs) | `IoPool` にマルチボリューム将来案の doc |

## 設計メモ

- **並列マルチドライブの帯域足し算は未モデル化**。`effective_secondary_storage_peak_bw` は「複数台プロファイルがあるときに遅い先頭エントリだけ拾う」バグを避けるため **SSD クラスの最大ピーク**を使うに留まる（実際の並列シャーディングではない）。
- 旧 JSON に `storage_pool_id` が無い場合は serde `default` で `0`。

## 検証

- `cargo check -p hypura`（`RUSTC_WRAPPER` 無効化推奨）。初回は `hypura-sys` の CMake で時間がかかる場合あり。

## なんJ風ひとこと

用語のすり替えで殴り合いにならないように README に書いといた。マルチ SSD の本丸はスケジューラと IoPool の二枚看板、いまはプール ID 置いとくだけでワラ。

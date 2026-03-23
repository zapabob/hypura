# _docs — Hypura 実装ドキュメント / Implementation Documentation

このディレクトリには Hypura の設計決定・実装詳細・移植記録が含まれています。

This directory contains design decisions, implementation details, and porting records for Hypura.

---

## ファイル一覧 / Files

| ファイル | 内容 |
|---|---|
| [implementation-log.md](./implementation-log.md) | 全実装の時系列ログ（スキャフォールドから Windows ポートまで）|
| [windows-wsl2-port.md](./windows-wsl2-port.md) | Windows/WSL2 + CUDA ポートの詳細設計ドキュメント |

---

## 目的 / Purpose

- 将来の開発者（および LLM）が設計決定の背景を理解できるようにする
- プラットフォーム固有の実装詳細を一箇所に集約する
- ポート作業のアーキテクチャ上の理由を記録する

These docs exist so future developers (and LLMs) can understand the *why* behind design decisions, have a single place for platform-specific implementation details, and trace the architectural rationale of porting work.

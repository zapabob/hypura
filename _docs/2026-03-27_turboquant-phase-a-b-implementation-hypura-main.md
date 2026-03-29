# TurboQuant Phase A-B 実装ログ

- 日付: 2026-03-27
- 対象リポジトリ: `hypura-main`
- 実装者: Codex
- テーマ: TurboQuant を GGUF weight format 変更ではなく、KV-cache runtime feature として Hypura に統合するための Phase A-B seam 実装

## 1. このログの目的

今回の変更は「TurboQuant の本格数理実装」ではなく、その前段となる実運用向けの配線と責務分離を入れることが目的。

ゴールは以下。

- `run` / `serve` / `bench` の全部で同じ TurboQuant モード解決を使う
- sidecar config の明示指定と自動発見を入れる
- `exact` をデフォルトとして維持する
- `paper-key-only` を最初の非デフォルト経路として追加する
- `paper-full-kv` と `research-kv-split` は将来用の明示的 placeholder にする
- 既存 exact runtime を壊さず、後続の math port が入りやすい seam を先に作る

## 2. 変更したファイル

### 新規追加

- `src/model/turboquant_sidecar.rs`
- `src/cache/kv_codec.rs`
- `_docs/2026-03-27_turboquant-phase-a-b-implementation-hypura-main.md`

### 既存変更

- `src/main.rs`
- `src/cli/run.rs`
- `src/cli/serve.rs`
- `src/cli/bench.rs`
- `src/compute/inference.rs`
- `src/model/mod.rs`
- `src/cache/mod.rs`
- `src/server/routes.rs`

補足:

- このログで扱っている TurboQuant core diff は上記の関連ファイル群に概ね閉じている。
- ただし repo の worktree 全体には TurboQuant 以外の未整理な変更も残っており、このログではそれらを別物として扱う。

## 3. 実装した内容

### 3.1 CLI surface の追加 (implemented)

`Run` / `Serve` / `Bench` に以下の引数を追加した。

- `--turboquant-mode <exact|paper-key-only|paper-full-kv|research-kv-split>`
- `--turboquant-config <path>`

デフォルトは `exact`。

この変更により、CLI 層だけでモードを決め打ちせず、全コマンドで同じ runtime resolver に委譲する形にした。

### 3.2 TurboQuant sidecar loader の追加 (implemented)

`src/model/turboquant_sidecar.rs` に以下を追加。

- `TurboQuantMode`
- `TurboQuantSchemaKind`
- `PaperTurboQuantConfig`
- `ResearchTurboQuantConfig`
- `TurboQuantSidecarConfig`
- `ResolvedTurboQuantConfig`
- `resolve_turboquant_config(...)`
- `auto_discover_sidecar_path(...)`
- `parse_sidecar_config(...)`

#### 仕様

- `exact` は sidecar 不要
- `paper-key-only` / `paper-full-kv` は `paper` schema 必須
- `research-kv-split` は `research` schema 必須
- non-`exact` で sidecar が無い場合は early fail
- schema が mode と一致しない場合も early fail

#### 自動発見規約

model path が `model.gguf` の場合:

- paper 系: `model.turboquant_config.paper.json`
- research 系: `model.turboquant_config.research.json`

Hypura 既存の `with_extension("permutations.json")` スタイルに寄せて、sidecar を model 近傍に置く前提を維持した。

### 3.3 sidecar と model metadata の軽い整合性チェック (implemented)

本格 schema validator はまだ入れていないが、実害の大きい mismatch を避けるために以下を optional に検証するようにした。

- `num_layers`
- `num_kv_heads`
- `head_dim`

sidecar にこれらのキーが存在する場合、`ModelMetadata` と一致しなければ fail する。

### 3.4 shared startup resolver の追加 (implemented)

`src/compute/inference.rs` に `RuntimeSetup` と `resolve_runtime_setup(...)` を追加した。

この resolver がまとめてやること:

1. hardware profile load / refresh
2. GGUF parse
3. `ModelMetadata` 抽出
4. TurboQuant mode + sidecar resolve
5. placement plan 計算
6. placement summary 計算
7. `n_gpu_layers` 決定

これで `run` / `serve` / `bench` が別々に GGUF/metadata/placement/turboquant を解決しなくて済むようにした。

### 3.5 inference 側への TurboQuant 情報の配線 (seam-only)

`LoadedModel` に以下を追加。

- `turboquant: ResolvedTurboQuantConfig`

さらに以下の関数シグネチャを更新した。

- `load_model(..., turboquant: &ResolvedTurboQuantConfig)`
- `generate_with_nvme_scheduling(..., turboquant: &ResolvedTurboQuantConfig)`

現時点では TurboQuant codec は llama.cpp の KV 実装に食い込ませていない。mode/config が runtime object と direct-generation path の両方に通るようになったのは事実だが、ここは runtime 接続の seam までで、実 KV 経路への介入はまだ行っていない。

### 3.6 KV codec seam の追加 (seam-only)

`src/cache/kv_codec.rs` に以下を追加。

- `KvCodec` trait
- `ExactKvCodec`
- `PaperKeyOnlyCodec`
- `PaperFullKvCodec`
- `build_kv_codec(...)`

#### 現時点の役割

- `ExactKvCodec`
  - exact path の placeholder-backed in-memory implementation
- `PaperKeyOnlyCodec`
  - K append/score は将来の TurboQuant 数理実装用 stub
  - V は exact で保持
- `PaperFullKvCodec`
  - type は定義
  - `new()` は明示的に `not implemented yet` を返す

#### 重要な意図

この codec 層は「今すぐ動く KV compression 実装」ではなく、「TurboQuant 数理実装を後から差し込める ownership seam」を作るためのもの。

`generate_with_nvme_scheduling()` と `load_model()` の入口で `build_kv_codec(...)` を呼び、mode が不正ならその場で fail するようにしてある。

### 3.7 run / serve / bench の統一 (implemented)

#### `run`

- shared runtime resolver を使うように変更
- startup 時に `TurboQuant: mode=..., schema=..., config=...` を表示
- direct `generate_with_nvme_scheduling()` path に `ResolvedTurboQuantConfig` を渡すように変更

#### `serve`

- shared runtime resolver を使うように変更
- `load_model()` に TurboQuant config を渡すように変更
- startup banner に TurboQuant 情報を表示

#### `bench`

- shared runtime resolver を使うように変更
- Hypura run 側に TurboQuant config を渡すように変更
- bench 出力に TurboQuant 情報を表示
- bench JSON config に以下を追加
  - `turboquant_mode`
  - `turboquant_schema`

### 3.8 server `/api/show` への露出 (implemented)

`src/server/routes.rs` の `AppState` に `turboquant` を追加し、`/api/show` の `model_info` に以下を追加した。

- `hypura.turboquant.mode`
- `hypura.turboquant.schema`
- `hypura.turboquant.config_path`

これで serving 中に active mode が見えるようになった。

## 4. 追加したテスト

### `src/model/turboquant_sidecar.rs`

- paper config parse
- schema/mode mismatch fail
- paper sidecar autodiscovery
- exact mode ignores sidecar
- non-exact missing sidecar fail

### `src/cache/kv_codec.rs`

- exact codec round-trip
- paper-key-only keeps V exact
- experimental mode reject

### `src/cli/bench.rs`

- bench config serialization に `turboquant_mode` / `turboquant_schema` が入ること

## 5. 今回 intentionally 未実装のもの (unimplemented)

今回の差分では以下はまだやっていない。

### 5.1 llama.cpp の実 KV path への介入 (unimplemented)

まだ以下は未実装。

- 実トークン append 時に llama.cpp から K/V を抜く
- `KvCodec` を attention scoring 実経路に入れる
- compressed K を復元または近似スコアに使う
- exact / compressed V 読み出しを attention output に反映する

理由:

- まず runtime seam と mode/config contract を先に固定したかったため
- 既存 exact inference を壊さずに段階導入したかったため

### 5.2 paper-faithful K math (unimplemented)

未実装。

- explicit norm
- Haar rotation
- coordinate-wise Lloyd-Max
- residual QJL estimator

これは次フェーズで `PaperKeyOnlyCodec` の中身として入れる前提。

### 5.3 full-KV / research path (unimplemented)

未実装。

- `paper-full-kv` は type を置いた上で明示的 fail
- `research-kv-split` も明示的 fail

silent fallback を避けるのが優先。

## 6. 検証結果

### 成功

- `cargo fmt` は実行済み
- 追加テストコードは配置済み
- この文書ファイル自体は UTF-8 で正常に読めており、以前見えた garbling は表示経路または文字コード解釈の依存だった
- TurboQuant core diff は主に関連ファイル群に閉じている一方、repo worktree 全体には TurboQuant 以外の未整理な変更も残っている

### 失敗した検証

`cargo test` はこの Windows 環境で最後まで通せなかった。

#### 観測した問題 1

Rust compile 中に以下のようなエラー。

- `cached cgu ... should have an object file, but doesn't`

これは source error ではなく build cache / `sccache` 系の問題に見える。

#### 観測した問題 2

fresh target dir を使っても、build script 実行時に以下で止まった。

- `アクセスが拒否されました。 (os error 5)`

対象は Cargo が生成した build-script executable。

推定:

- この環境では Rust build artifacts の実行が制限されている
- そのため project compile failure の前に環境制約で停止している

## 7. 次の担当者への引き継ぎ

次にやるべきことは以下。

### 優先度高

1. `cargo test` / `cargo check` を build-script 実行可能な環境で再実行
2. TurboQuant sidecar JSON schema の実サンプルを用意
3. `paper-key-only` の数理実装を `PaperKeyOnlyCodec` に入れる

### その次

4. llama.cpp / FFI 側の KV capture and score seam を探す
5. `KvCodec` を実 runtime path に接続する
6. bench で `exact` vs `paper-key-only` の比較を保存する

### 注意点

- `exact` を default のまま維持すること
- sidecar failure 時に exact fallback しないこと
- `paper-full-kv` を default にしないこと
- research schema と paper schema を混ぜないこと

## 8. 実装上の判断メモ

### 判断 1: sidecar は GGUF に埋め込まない

今回の段階で GGUF metadata に押し込まず、JSON sidecar に分離した。

理由:

- iteration が速い
- rollback が容易
- runtime codec と weight container の責務を混ぜない

### 判断 2: `run` / `bench` も `serve` と同じ resolver を使う

最初は `serve` だけ `load_model()` 経由、`run` / `bench` は direct generation なので乖離しやすい構造だった。

これを shared `resolve_runtime_setup(...)` で寄せた。

### 判断 3: 先に trait を置く

codec の実装はまだ弱いが、trait と mode-based factory を先に固定した。

理由:

- 後続作業で ownership の大きな再分割を避けられる
- `paper-key-only` / `paper-full-kv` / `research-kv-split` を型として先に宣言できる

## 9. 現時点の実質ステータス

今回の差分は「TurboQuant を本当に使って推論が速くなる」段階ではない。

今の状態は以下。

- mode/config plumbing: あり
- sidecar discovery/validation: あり
- server/bench visibility: あり
- codec seam: あり
- exact default preserved: あり
- paper-key-only runtime connection: まだ
- paper-key-only runtime math: まだ
- paper full-KV runtime math: まだ
- research codec runtime math: まだ
- repo worktree 全体の unrelated modifications: あり

要するに、Phase A は実装済み、Phase B は runtime seam と部分配線まで進んだが、実 KV 接続と数理本体は次段階である。TurboQuant core diff は狭いが、worktree 全体はそれより広く変更されている。

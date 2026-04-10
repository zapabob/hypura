# llama.cpp 公式取り込み / zapabob 独自機能 調査ログ

- **日時 (UTC):** 2026-04-05（スクリプト実行時 `datetime.now(timezone.utc)`）
- **worktree:** hypura-main
- **サブモジュール:** `vendor/llama.cpp`（`origin` = ggerganov/llama.cpp、`zapabob` = zapabob/llama.cpp）

## なんJ風まとめ（雑コラム）

公式 master は取り込み済みで、いまの HEAD は「公式の先端 + TurboQuant だけ乗っかってる」状態っぽい。取り残しコミットはゼロなので「あとは公式が進んだらマージし直せ」って感じ。独自部分は11ファイルに閉じてるから、コンフlict来てもそこさえ守れば大惨事にはなりにくいんご。

## CoT 仮説検証

1. **仮説:** `vendor/llama.cpp` は公式 master より遅れている。
   - **検証:** `merge-base --is-ancestor origin/master HEAD` → True。
   - **結論:** 公式 master の先端は既に HEAD に含まれる。未取り込みの upstream コミットは 0。

2. **仮説:** 独自機能はファイル単位で分離できる。
   - **検証:** `git diff --name-only origin/master...HEAD` → 11 ファイル（turboquant / kv-cache / convert / cmake / llama-graph）。
   - **結論:** 「公式と同等の機能が出たら公式版を採用し、差分はこの11ファイル相当を手で当て直す」方針が取りやすい。

3. **仮説:** `git diff HEAD..upstream` で incoming ファイルが取れる。
   - **検証:** ツリー差分だとフォーク側の変更が混ざる。
   - **修正:** `git log HEAD..upstream --name-only` に変更済み。

## 使い方（Python / tqdm）

```powershell
cd c:\Users\downl\Desktop\hypura-main\hypura-main
py -3 scripts\merge_llama_cpp_upstream.py --survey-only --json-out _docs\llama_cpp_merge_survey.json
```

- マージまで行く場合（作業ブランチ `codex/triality-defaults`）:

```powershell
py -3 scripts\merge_llama_cpp_upstream.py --survey --merge-message "merge: upstream; keep Triality/TurboQuant"
```

- コンフリクト時、CI 用ファイルだけ公式側: `--theirs-paths '.github/**'`

## hypura-sys / API 追従メモ

- `hypura-sys/wrapper.h` は `llama.h` / `ggml*.h` に依存。`include/` や `ggml/include/*.h` が incoming に出たら `cargo build -p hypura-sys` で bindgen を再確認。
- TurboQuant は主に `src/llama-turboquant.*` と `llama-kv-cache.*`。**PROTECTED** 指定は `scripts/merge_llama_cpp_upstream.py` の `PROTECTED_PATHS` を参照。

## 成果物

- スクリプト: `scripts/merge_llama_cpp_upstream.py`（`--survey` / `--survey-only` / `--json-out` / `--security-scan`）
- スナップショット JSON: `_docs/llama_cpp_merge_survey.json`（再生成可）

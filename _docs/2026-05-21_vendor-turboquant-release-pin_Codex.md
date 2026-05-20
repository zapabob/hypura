# Overview

Updated Hypura's vendored `turboquant-cuda` submodule pointer to a
clone-reproducible Turboquant-CUDA commit that also commits its nested
`zapabob/llama.cpp` runtime pointer.

# Background / Requirements

- Hypura is the product runtime/distribution layer for the zapabob
  Turboquant-CUDA + llama.cpp + Hypura stack.
- A release pin should not depend on an uncommitted nested submodule checkout
  inside `vendor/turboquant-cuda`.
- The vendored Turboquant-CUDA tree should point at a commit that contains its
  own nested `zapabob/llama.cpp` pointer.

# Assumptions / Decisions

- The target Turboquant-CUDA commit is
  `eac621fd8b5e0af8a10596c1feb0863a34f08cd8`.
- That commit pins nested `zapabob/llama.cpp` to
  `a020899959e5ad83ace83fca3042b6a47b7153c4`, matching the parent Triality
  runtime validation line.
- The old `vendor/turboquant-cuda/vendor/llama.cpp` checkout was a clean
  leftover from the older submodule layout and was removed after the switch to
  the `zapabob/llama.cpp` layout.

# Changed Files

- `vendor/turboquant-cuda`
- `_docs/2026-05-21_vendor-turboquant-release-pin_Codex.md`

# Implementation Details

- Fetched `origin/codex/dual-family-readme-refresh` inside
  `vendor/turboquant-cuda`.
- Checked out `eac621fd8b5e0af8a10596c1feb0863a34f08cd8`.
- Ran recursive submodule sync/update for the new nested `zapabob/llama.cpp`
  path.
- Verified the old nested `vendor/llama.cpp` checkout was clean before removing
  the untracked leftover directory.

# Commands Run

```powershell
git -C vendor/turboquant-cuda fetch origin codex/dual-family-readme-refresh
git -C vendor/turboquant-cuda checkout eac621fd8b5e0af8a10596c1feb0863a34f08cd8
git -C vendor/turboquant-cuda submodule sync --recursive
git -C vendor/turboquant-cuda submodule update --init --recursive
git -C vendor/turboquant-cuda/vendor/llama.cpp status --short --branch
Remove-Item -LiteralPath vendor\turboquant-cuda\vendor -Recurse -Force
git submodule status --recursive
```

# Test / Verification Results

- `vendor/turboquant-cuda` now resolves to
  `eac621fd8b5e0af8a10596c1feb0863a34f08cd8`.
- Nested `vendor/turboquant-cuda/zapabob/llama.cpp` now resolves to
  `a020899959e5ad83ace83fca3042b6a47b7153c4`.
- `git -C vendor/turboquant-cuda status --ignore-submodules=none --short
  --branch` reports a clean detached checkout.

# Residual Risks

- Full runtime verification is expected at the Triality parent level after the
  parent submodule pins are updated.

# Recommended Next Actions

- Commit and push this Hypura branch.
- Update the Triality parent repo's `repos/hypura` and `repos/Turboquant-CUDA`
  pins to the newly committed child SHAs.

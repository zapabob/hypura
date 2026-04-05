#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate merging ggerganov/llama.cpp master into vendor/llama.cpp (fork branch).

Uses Git for the actual merge; this script runs fetch, optional inventory, merge,
optional conflict resolution (--theirs-paths), and post-merge verification.

Caption: Git-wrapped upstream merge with tqdm phase progress for llama.cpp fork.
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LLAMA_DIR = REPO_ROOT / "vendor" / "llama.cpp"

# Paths that must exist and retain TurboQuant / Triality integration markers.
PROTECTED_PATHS = (
    "convert_hf_to_gguf.py",
    "src/llama-turboquant.cpp",
    "src/llama-turboquant.h",
    "src/llama-kv-cache.cpp",
    "src/llama-kv-cache.h",
)

PROTECTED_SUBSTRINGS = (
    ("convert_hf_to_gguf.py", "hypura.turboquant"),
    ("src/llama-kv-cache.h", "llama-turboquant.h"),
)


def _tqdm_phases(desc: str, phases: list[str]):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(phases, desc=desc, unit="phase", ascii=True)
    except Exception:
        return phases


def _git(
    repo: Path,
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = ["git", "-C", str(repo), *args]
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


def _unmerged_paths(repo: Path) -> list[str]:
    r = _git(
        repo,
        ["diff", "--name-only", "--diff-filter=U"],
        check=False,
    )
    if r.returncode != 0:
        return []
    return [p for p in r.stdout.splitlines() if p.strip()]


def _matches_any_glob(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path.replace("\\", "/"), pat) for pat in patterns)


def _inventory(repo: Path, upstream: str) -> None:
    print(f"\n=== Commits on HEAD not in {upstream} ===\n")
    r = _git(repo, ["log", "--oneline", f"{upstream}..HEAD"])
    print(r.stdout or "(none)")
    print(f"\n=== diff --stat {upstream}...HEAD (last 40 lines) ===\n")
    r2 = _git(repo, ["diff", "--stat", f"{upstream}...HEAD"])
    lines = (r2.stdout or "").splitlines()
    print("\n".join(lines[-40:]))


def _verify(repo: Path) -> None:
    missing = [p for p in PROTECTED_PATHS if not (repo / p).is_file()]
    if missing:
        raise SystemExit(f"verify failed: missing files: {missing}")

    for rel, needle in PROTECTED_SUBSTRINGS:
        text = (repo / rel).read_text(encoding="utf-8", errors="replace")
        if needle not in text:
            raise SystemExit(f"verify failed: {rel!r} missing substring {needle!r}")


def _apply_theirs_paths(
    repo: Path,
    patterns: list[str],
    *,
    allow_protected: bool,
) -> None:
    conflicts = _unmerged_paths(repo)
    if not conflicts:
        print("no unmerged paths; nothing to do for --theirs-paths")
        return

    for path in conflicts:
        norm = path.replace("\\", "/")
        if not _matches_any_glob(norm, patterns):
            continue
        if not allow_protected and norm in PROTECTED_PATHS:
            print(
                f"skip --theirs (protected): {path}; resolve manually or use "
                "--allow-theirs-protected",
                file=sys.stderr,
            )
            continue
        print(f"checkout --theirs -- {path}")
        _git(repo, ["checkout", "--theirs", "--", path])
        _git(repo, ["add", "--", path])

    remaining = _unmerged_paths(repo)
    if remaining:
        raise SystemExit(
            "merge still has conflicts:\n  " + "\n  ".join(remaining) + "\n"
            "Resolve manually, then git add && git commit, or git merge --abort."
        )

    print("All conflicts resolved via --theirs-paths; completing merge commit.")
    _git(repo, ["commit", "--no-edit"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge upstream llama.cpp master into local fork checkout.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=DEFAULT_LLAMA_DIR,
        help=f"path to llama.cpp clone (default: {DEFAULT_LLAMA_DIR})",
    )
    parser.add_argument(
        "--branch",
        default="codex/triality-defaults",
        help="local branch to checkout before merge",
    )
    parser.add_argument(
        "--upstream-remote",
        default="origin",
        help="remote name for ggerganov/llama.cpp",
    )
    parser.add_argument(
        "--upstream-ref",
        default="master",
        help="upstream branch name on upstream-remote",
    )
    parser.add_argument(
        "--fork-remote",
        default="zapabob",
        help="optional second remote (fork); fetched unless --no-fetch-fork",
    )
    parser.add_argument(
        "--merge-message",
        default="merge: upstream ggerganov/llama.cpp master; keep Triality/TurboQuant",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch + inventory only; no checkout/merge",
    )
    parser.add_argument(
        "--theirs-paths",
        nargs="*",
        default=[],
        metavar="GLOB",
        help=(
            "after a conflicted merge: git checkout --theirs for unmerged paths "
            "matching these globs (e.g. '.github/**')"
        ),
    )
    parser.add_argument(
        "--allow-theirs-protected",
        action="store_true",
        help="allow --theirs-paths to touch PROTECTED_PATHS (dangerous)",
    )
    parser.add_argument(
        "--no-fetch-fork",
        action="store_true",
        help="skip git fetch on fork-remote",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="fetch + inventory + verify only (no merge)",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not repo.is_dir():
        raise SystemExit(f"repo not found: {repo}")

    upstream = f"{args.upstream_remote}/{args.upstream_ref}"

    steps: list[str] = ["fetch_upstream"]
    if not args.no_fetch_fork:
        steps.append("fetch_fork")
    steps.append("inventory")
    if not args.dry_run:
        steps.append("checkout")
        if not args.skip_merge:
            steps.append("merge")
        steps.append("verify")

    for _step in _tqdm_phases("llama.cpp upstream", steps):
        if _step == "fetch_upstream":
            _git(repo, ["fetch", args.upstream_remote, args.upstream_ref])
        elif _step == "fetch_fork":
            _git(repo, ["fetch", args.fork_remote])
        elif _step == "inventory":
            _inventory(repo, upstream)
        elif _step == "checkout":
            _git(repo, ["checkout", args.branch])
        elif _step == "merge":
            r = _git(
                repo,
                ["merge", upstream, "-m", args.merge_message],
                check=False,
            )
            if r.returncode != 0:
                if _unmerged_paths(repo):
                    print(r.stderr or r.stdout, file=sys.stderr)
                    if args.theirs_paths:
                        _apply_theirs_paths(
                            repo,
                            args.theirs_paths,
                            allow_protected=args.allow_theirs_protected,
                        )
                    else:
                        raise SystemExit(
                            "merge conflict. Resolve manually or re-run with "
                            "--theirs-paths GLOB [...] for non-protected files."
                        )
                elif "Already up to date" in (r.stdout or ""):
                    print(r.stdout.strip())
                else:
                    print(r.stderr or r.stdout, file=sys.stderr)
                    raise SystemExit(r.returncode)
            else:
                out = (r.stdout or "").strip()
                if out:
                    print(out)
        elif _step == "verify":
            _verify(repo)

    if args.dry_run:
        print("\n(dry-run: stopped before checkout/merge)")

    ahead = _git(
        repo,
        ["rev-list", "--count", f"{upstream}..HEAD"],
        check=False,
    )
    behind = _git(
        repo,
        ["rev-list", "--count", f"HEAD..{upstream}"],
        check=False,
    )
    if ahead.returncode == 0 and behind.returncode == 0:
        print(
            f"\nSummary: HEAD vs {upstream}: "
            f"{ahead.stdout.strip()} ahead, {behind.stdout.strip()} behind"
        )


if __name__ == "__main__":
    main()

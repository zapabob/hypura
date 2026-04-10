#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate merging ggml-org/ggerganov llama.cpp master into vendor/llama.cpp (zapabob fork).

Uses Git for the actual merge; this script runs fetch, optional inventory, merge,
optional conflict resolution (--theirs-paths), post-merge verification, and
--survey / --json-out for Triality/TurboQuant maintenance.

Workflow when upstream adds an overlapping feature: prefer upstream implementation,
then re-apply only zapabob deltas (SO8/triality hooks, GGUF metadata, KV paths)
on the updated API — use ``--survey`` to list fork-only files and incoming
upstream paths that touch the public C API (hypura-sys bindgen).

Caption: Git-wrapped upstream merge with tqdm phase progress for llama.cpp fork.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
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

MERGE_POLICY = {
    "protected": {
        "convert_hf_to_gguf.py": "Keep Hypura GGUF TurboQuant metadata embedding and default knobs.",
        "src/llama-turboquant.cpp": "Keep Triality/SO8 runtime helpers unless upstream fully supersedes them.",
        "src/llama-turboquant.h": "Keep Triality/SO8 runtime interface unless upstream fully supersedes it.",
        "src/llama-kv-cache.cpp": "Keep KV-cache TurboQuant hooks and env bridge integration.",
        "src/llama-kv-cache.h": "Keep KV-cache TurboQuant wiring and runtime config exposure.",
    },
    "prefer_upstream": {
        "CMakeLists.txt": "Prefer upstream build plumbing when equivalent functionality exists.",
        "src/CMakeLists.txt": "Prefer upstream build plumbing when equivalent functionality exists.",
        "src/llama-graph.cpp": "Prefer upstream graph/runtime changes, then re-apply only TurboQuant benefits.",
        "tools/CMakeLists.txt": "Prefer upstream tool wiring when equivalent functionality exists.",
    },
    "reinject_benefits": {
        "convert_hf_to_gguf.py": "Re-apply Hypura GGUF metadata defaults after upstream converter changes.",
        "src/llama-graph.cpp": "Re-apply TurboQuant runtime hooks on top of upstream graph updates.",
        "src/llama-kv-cache.cpp": "Re-apply KV-cache Triality/SO8 behavior on top of upstream bugfixes.",
        "src/llama-kv-cache.h": "Re-apply KV-cache interface benefits on top of upstream API changes.",
    },
}

PROTECTED_SUBSTRINGS = (
    ("convert_hf_to_gguf.py", "hypura.turboquant"),
    ("src/llama-kv-cache.h", "llama-turboquant.h"),
)

DEFAULT_ALIGNMENT_CHECKS = (
    {
        "path": REPO_ROOT / "vendor" / "llama.cpp" / "convert_hf_to_gguf.py",
        "label": "GGUF outtype default",
        "needle": 'default="q8_0"',
    },
    {
        "path": REPO_ROOT / "vendor" / "llama.cpp" / "convert_hf_to_gguf.py",
        "label": "GGUF TurboQuant mode default",
        "needle": 'default="research-kv-split"',
    },
    {
        "path": REPO_ROOT / "vendor" / "llama.cpp" / "convert_hf_to_gguf.py",
        "label": "GGUF TrialitySO8 rotation default",
        "needle": 'default="triality_vector"',
    },
    {
        "path": REPO_ROOT / "src" / "main.rs",
        "label": "CLI TurboQuant mode default",
        "needle": "default_value_t = TurboQuantMode::ResearchKvSplit",
    },
    {
        "path": REPO_ROOT / "src" / "main.rs",
        "label": "CLI TrialitySO8 rotation default",
        "needle": "default_value_t = RotationPolicy::TrialityVector",
    },
)

# If these change in commits you are about to merge, rebuild hypura-sys / review bindgen.
API_TOUCH_PATHS = (
    "include/llama.h",
    "include/llama-cpp.h",
    "ggml/include/ggml.h",
    "ggml/include/ggml-backend.h",
    "ggml/include/ggml-alloc.h",
    "src/llama.cpp",
    "src/llama-graph.cpp",
    "src/llama-kv-cache.h",
    "src/llama-kv-cache.cpp",
)

_SECURITY_SUBJECT_RE = re.compile(
    r"security|cve|overflow|underflow|buffer|vulnerable|sanitizer|"
    r"oob|use-?after-?free|uaf|denial|crash\s*fix",
    re.IGNORECASE,
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


def _lines(cmd_out: str) -> list[str]:
    return [x.strip() for x in (cmd_out or "").splitlines() if x.strip()]


def _fork_only_files(repo: Path, upstream: str) -> list[str]:
    r = _git(repo, ["diff", "--name-only", f"{upstream}...HEAD"])
    return sorted(set(_lines(r.stdout)))


def _incoming_upstream_files(repo: Path, upstream: str) -> list[str]:
    """Files touched only by commits reachable from upstream but not from HEAD."""
    r = _git(
        repo,
        [
            "log",
            f"HEAD..{upstream}",
            "--pretty=format:",
            "--name-only",
        ],
        check=False,
    )
    return sorted(set(_lines(r.stdout)))


def _incoming_upstream_commits(repo: Path, upstream: str, limit: int) -> list[str]:
    r = _git(
        repo,
        ["log", "--oneline", f"HEAD..{upstream}", f"-n{max(1, limit)}"],
        check=False,
    )
    return _lines(r.stdout)


def _upstream_contains_head(repo: Path, upstream: str) -> bool:
    r = _git(
        repo,
        ["merge-base", "--is-ancestor", "HEAD", upstream],
        check=False,
    )
    return r.returncode == 0


def _head_contains_upstream(repo: Path, upstream: str) -> bool:
    r = _git(
        repo,
        ["merge-base", "--is-ancestor", upstream, "HEAD"],
        check=False,
    )
    return r.returncode == 0


def _filter_api_paths(paths: list[str]) -> list[str]:
    norm = [p.replace("\\", "/") for p in paths]
    out: list[str] = []
    for p in sorted(set(norm)):
        if p in API_TOUCH_PATHS:
            out.append(p)
        elif (p.startswith("include/") or p.startswith("ggml/include/")) and p.endswith(
            ".h"
        ):
            out.append(p)
    return out


def _security_flagged_commits(repo: Path, upstream: str, scan_limit: int) -> list[str]:
    raw = _incoming_upstream_commits(repo, upstream, scan_limit * 3)
    out: list[str] = []
    for line in raw:
        if _SECURITY_SUBJECT_RE.search(line):
            out.append(line)
        if len(out) >= scan_limit:
            break
    return out


def _normalized_policy_paths(bucket: str) -> list[str]:
    return sorted(MERGE_POLICY[bucket].keys())


def _classify_merge_policy(path: str) -> list[str]:
    norm = path.replace("\\", "/")
    buckets: list[str] = []
    for bucket, mapping in MERGE_POLICY.items():
        if norm in mapping:
            buckets.append(bucket)
    return buckets


def _policy_overlap(paths: list[str]) -> dict[str, list[str]]:
    overlaps = {bucket: [] for bucket in MERGE_POLICY}
    for path in sorted(set(p.replace("\\", "/") for p in paths)):
        for bucket in _classify_merge_policy(path):
            overlaps[bucket].append(path)
    return overlaps


def _default_alignment_status() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for check in DEFAULT_ALIGNMENT_CHECKS:
        path = Path(check["path"])
        exists = path.is_file()
        text = path.read_text(encoding="utf-8", errors="replace") if exists else ""
        needle = str(check["needle"])
        out.append(
            {
                "label": check["label"],
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "expected": needle,
                "ok": exists and needle in text,
            }
        )
    return out


def _build_survey_dict(repo: Path, upstream: str, *, security_scan: int) -> dict:
    fork_only = _fork_only_files(repo, upstream)
    incoming_files = _incoming_upstream_files(repo, upstream)
    fork_overlap = _policy_overlap(fork_only)
    incoming_overlap = _policy_overlap(incoming_files)
    default_alignment = _default_alignment_status()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_ref": upstream,
        "head_short": _lines(_git(repo, ["rev-parse", "--short", "HEAD"]).stdout)[0],
        "upstream_short": _lines(_git(repo, ["rev-parse", "--short", upstream]).stdout)[0],
        "head_contains_upstream": _head_contains_upstream(repo, upstream),
        "upstream_contains_head": _upstream_contains_head(repo, upstream),
        "fork_only_commits": _lines(
            _git(repo, ["log", "--oneline", f"{upstream}..HEAD"]).stdout
        ),
        "fork_only_files": fork_only,
        "incoming_upstream_commits_preview": _incoming_upstream_commits(
            repo, upstream, 25
        ),
        "incoming_upstream_files": incoming_files,
        "incoming_api_touch_paths": _filter_api_paths(incoming_files),
        "security_flagged_incoming_commits": _security_flagged_commits(
            repo, upstream, security_scan
        ),
        "protected_paths_ok": all((repo / p).is_file() for p in PROTECTED_PATHS),
        "merge_policy": {
            bucket: [
                {"path": path, "reason": reason}
                for path, reason in sorted(mapping.items())
            ]
            for bucket, mapping in MERGE_POLICY.items()
        },
        "fork_policy_overlap": fork_overlap,
        "incoming_policy_overlap": incoming_overlap,
        "default_alignment": default_alignment,
        "default_alignment_ok": all(item["ok"] for item in default_alignment),
    }


def _print_survey(repo: Path, upstream: str, *, security_scan: int) -> dict:
    data = _build_survey_dict(repo, upstream, security_scan=security_scan)
    print("\n=== Survey (fork vs upstream) ===\n")
    print(f"HEAD {data['head_short']} | upstream {data['upstream_short']}")
    print(f"HEAD contains upstream: {data['head_contains_upstream']}")
    print(f"upstream contains HEAD: {data['upstream_contains_head']}")
    print(f"\n--- Fork-only commits ({len(data['fork_only_commits'])}) ---")
    for c in data["fork_only_commits"]:
        print(c)
    print(f"\n--- Fork-only files ({len(data['fork_only_files'])}) ---")
    for f in data["fork_only_files"]:
        print(f"  {f}")
    inc = data["incoming_upstream_commits_preview"]
    print(f"\n--- Incoming upstream commits (preview, {len(inc)}) ---")
    if inc:
        for c in inc:
            print(c)
    else:
        print("(none - already merged or same tip)")
    api = data["incoming_api_touch_paths"]
    print(f"\n--- Incoming paths that may require hypura-sys / C API review ---")
    if api:
        for p in api:
            print(f"  {p}")
    else:
        print("(none in current incoming file list)")
    print(f"\n--- Merge policy overlaps: fork-only ---")
    for bucket in ("protected", "prefer_upstream", "reinject_benefits"):
        hits = data["fork_policy_overlap"][bucket]
        print(f"{bucket}: {len(hits)}")
        for path in hits:
            print(f"  {path}")
    print(f"\n--- Merge policy overlaps: incoming upstream ---")
    for bucket in ("protected", "prefer_upstream", "reinject_benefits"):
        hits = data["incoming_policy_overlap"][bucket]
        print(f"{bucket}: {len(hits)}")
        for path in hits:
            print(f"  {path}")
    sec = data["security_flagged_incoming_commits"]
    print(
        f"\n--- Incoming commits with security-ish subjects (heuristic, max {security_scan}) ---"
    )
    if sec:
        for c in sec:
            print(c)
    else:
        print("(none matched)")
    print(f"\n--- TrialitySO8 default alignment ---")
    for item in data["default_alignment"]:
        status = "ok" if item["ok"] else "MISMATCH"
        print(f"[{status}] {item['label']} -> {item['path']}")
    print(f"\nprotected_paths_ok: {data['protected_paths_ok']}")
    print(f"default_alignment_ok: {data['default_alignment_ok']}")
    return data


def _verify(repo: Path) -> None:
    missing = [p for p in PROTECTED_PATHS if not (repo / p).is_file()]
    if missing:
        raise SystemExit(f"verify failed: missing files: {missing}")

    for rel, needle in PROTECTED_SUBSTRINGS:
        text = (repo / rel).read_text(encoding="utf-8", errors="replace")
        if needle not in text:
            raise SystemExit(f"verify failed: {rel!r} missing substring {needle!r}")

    misaligned = [
        item for item in _default_alignment_status() if not item["ok"]
    ]
    if misaligned:
        details = "\n".join(
            f"  {item['label']}: {item['path']} missing {item['expected']!r}"
            for item in misaligned
        )
        raise SystemExit(f"verify failed: TrialitySO8 defaults misaligned:\n{details}")


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
    parser.add_argument(
        "--survey",
        action="store_true",
        help="print fork vs upstream survey (incoming API paths, fork-only files)",
    )
    parser.add_argument(
        "--survey-only",
        action="store_true",
        help="fetch + survey + optional --json-out, then exit (no checkout/merge/verify)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="write survey JSON to this path (UTF-8)",
    )
    parser.add_argument(
        "--security-scan",
        type=int,
        default=20,
        metavar="N",
        help="max incoming commits to list with security-ish subjects (heuristic)",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not repo.is_dir():
        raise SystemExit(f"repo not found: {repo}")

    upstream = f"{args.upstream_remote}/{args.upstream_ref}"

    def _write_json_survey(data: dict) -> None:
        if args.json_out is None:
            return
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote survey JSON -> {args.json_out}")

    if args.survey_only:
        steps = ["fetch_upstream"]
        if not args.no_fetch_fork:
            steps.append("fetch_fork")
        steps.append("survey")
        for _step in _tqdm_phases("llama.cpp survey", steps):
            if _step == "fetch_upstream":
                _git(repo, ["fetch", args.upstream_remote, args.upstream_ref])
            elif _step == "fetch_fork":
                _git(repo, ["fetch", args.fork_remote])
            elif _step == "survey":
                sdata = _print_survey(
                    repo, upstream, security_scan=args.security_scan
                )
                _write_json_survey(sdata)
        return

    steps: list[str] = ["fetch_upstream"]
    if not args.no_fetch_fork:
        steps.append("fetch_fork")
    steps.append("inventory")
    if args.survey:
        steps.append("survey")
    if not args.dry_run:
        steps.append("checkout")
        if not args.skip_merge:
            steps.append("merge")
        steps.append("verify")

    survey_data: dict | None = None
    for _step in _tqdm_phases("llama.cpp upstream", steps):
        if _step == "fetch_upstream":
            _git(repo, ["fetch", args.upstream_remote, args.upstream_ref])
        elif _step == "fetch_fork":
            _git(repo, ["fetch", args.fork_remote])
        elif _step == "inventory":
            _inventory(repo, upstream)
        elif _step == "survey":
            survey_data = _print_survey(
                repo, upstream, security_scan=args.security_scan
            )
            _write_json_survey(survey_data)
        elif _step == "checkout":
            _git(repo, ["checkout", args.branch])
        elif _step == "merge":
            if _head_contains_upstream(repo, upstream):
                print(
                    f"\nmerge skipped: HEAD already contains {upstream} "
                    "(fast-forward not needed)."
                )
            else:
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

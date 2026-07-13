from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


SEMVER_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-((?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


def semantic_version(value: str) -> str:
    if SEMVER_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(f"version is not strict SemVer: {value}")
    return value


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result


def submodule_entries(repo: Path, commit: str) -> list[tuple[str, str, str]]:
    blob = f"{commit}:.gitmodules"
    configured = git(
        repo,
        "config",
        "--blob",
        blob,
        "--get-regexp",
        r"^submodule\..*\.path$",
        check=False,
    )
    if configured.returncode == 1:
        return []
    if configured.returncode != 0:
        raise RuntimeError(configured.stderr.strip())

    entries: list[tuple[str, str, str]] = []
    for line in configured.stdout.splitlines():
        key, path = line.split(maxsplit=1)
        tree = git(repo, "ls-tree", commit, "--", path).stdout.strip()
        if not tree:
            raise RuntimeError(f"missing gitlink for {path} at {commit}")
        sha = tree.split(maxsplit=3)[2]
        url_key = f"{key[:-5]}.url"
        url = git(repo, "config", "--blob", blob, "--get", url_key).stdout.strip()
        entries.append((path, sha, url))
    return entries


def validated_submodule_path(value: str, destination: Path) -> Path:
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or relative.drive
        or relative == Path(".")
        or ".." in relative.parts
    ):
        raise RuntimeError(f"unsafe submodule path: {value}")
    resolved_root = destination.resolve()
    resolved_path = (destination / relative).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise RuntimeError(f"submodule path escapes staging root: {value}")
    return relative


def extract_archive(repo: Path, commit: str, destination: Path) -> dict[str, int]:
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", commit],
        check=True,
        capture_output=True,
    ).stdout
    modes: dict[str, int] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        for member in source.getmembers():
            modes[member.name.rstrip("/")] = member.mode
        source.extractall(destination, filter="data")
    return modes


def materialize(
    repo: Path,
    commit: str,
    destination: Path,
    relative_root: Path,
    manifest: list[dict[str, str]],
    modes: dict[str, int],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    local_modes = extract_archive(repo, commit, destination)
    for name, mode in local_modes.items():
        modes[(relative_root / name).as_posix()] = mode

    for path, sha, url in submodule_entries(repo, commit):
        relative_path = validated_submodule_path(path, destination)
        submodule_repo = repo / relative_path
        if not submodule_repo.exists():
            raise RuntimeError(f"submodule is not initialized: {submodule_repo}")
        git(submodule_repo, "cat-file", "-e", f"{sha}^{{commit}}")
        submodule_destination = destination / relative_path
        if submodule_destination.exists():
            if submodule_destination.is_dir():
                shutil.rmtree(submodule_destination)
            else:
                submodule_destination.unlink()
        submodule_relative = relative_root / relative_path
        manifest.append(
            {"path": submodule_relative.as_posix(), "sha": sha, "url": url}
        )
        materialize(
            submodule_repo,
            sha,
            submodule_destination,
            submodule_relative,
            manifest,
            modes,
        )


def deterministic_tarball(
    staging: Path,
    output: Path,
    archive_root: str,
    modes: dict[str, int],
) -> None:
    paths = sorted(staging.rglob("*"), key=lambda path: path.relative_to(staging).as_posix())
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive:
                root_info = tarfile.TarInfo(archive_root)
                root_info.type = tarfile.DIRTYPE
                root_info.mode = 0o755
                root_info.mtime = 0
                archive.addfile(root_info)
                for path in paths:
                    relative = path.relative_to(staging).as_posix()
                    info = archive.gettarinfo(path, f"{archive_root}/{relative}")
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = modes.get(relative, info.mode)
                    if info.isfile():
                        with path.open("rb") as payload:
                            archive.addfile(info, payload)
                    else:
                        archive.addfile(info)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--version", type=semantic_version, required=True)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    commit = git(repo, "rev-parse", f"{args.commit}^{{commit}}").stdout.strip()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"hypura-v{args.version}-fullsource.tar.gz"

    manifest: list[dict[str, str]] = []
    modes: dict[str, int] = {}
    with tempfile.TemporaryDirectory(dir=output_dir, prefix="fullsource-") as temporary:
        staging = Path(temporary) / "source"
        materialize(repo, commit, staging, Path(), manifest, modes)
        source_manifest = {
            "format": 1,
            "hypura_sha": commit,
            "submodules": sorted(manifest, key=lambda item: item["path"]),
            "version": args.version,
        }
        manifest_path = staging / "SOURCE-MANIFEST.json"
        manifest_path.write_text(
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        modes["SOURCE-MANIFEST.json"] = 0o644
        required = [
            staging / "Cargo.toml",
            staging / "vendor/llama.cpp/CMakeLists.txt",
            staging / "vendor/turboquant-cuda/README.md",
            staging / "vendor/turboquant-cuda/zapabob/llama.cpp/CMakeLists.txt",
        ]
        missing = [str(path.relative_to(staging)) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"fullsource is missing required paths: {', '.join(missing)}")
        required_submodules = {
            "vendor/llama.cpp",
            "vendor/turboquant-cuda",
            "vendor/turboquant-cuda/zapabob/llama.cpp",
        }
        materialized_submodules = {entry["path"] for entry in manifest}
        missing_submodules = sorted(required_submodules - materialized_submodules)
        if missing_submodules:
            raise RuntimeError(
                "fullsource manifest is missing required submodules: "
                + ", ".join(missing_submodules)
            )
        deterministic_tarball(staging, output, f"hypura-v{args.version}", modes)

    checksum = sha256(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="ascii", newline="\n")
    print(output)
    print(checksum_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def existing_file(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=semantic_version, required=True)
    parser.add_argument("--channel", choices=("stable", "main"), required=True)
    parser.add_argument("--cli", type=existing_file, required=True)
    parser.add_argument("--fullsource", type=existing_file, required=True)
    parser.add_argument("--msi", type=existing_file, required=True)
    parser.add_argument("--nsis", type=existing_file, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def asset_names(version: str, channel: str) -> dict[str, str]:
    prefix = "hypura-main" if channel == "main" else "hypura"
    desktop = "hypura-desktop-main" if channel == "main" else "hypura-desktop"
    return {
        "cli": f"{prefix}-v{version}-windows-x86_64-sm120.exe",
        "fullsource": f"{prefix}-v{version}-fullsource.tar.gz",
        "msi": f"{desktop}-v{version}-windows-x64.msi",
        "nsis": f"{desktop}-v{version}-windows-x64-setup.exe",
    }


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = asset_names(args.version, args.channel)
    sources = {
        "cli": args.cli,
        "fullsource": args.fullsource,
        "msi": args.msi,
        "nsis": args.nsis,
    }
    staged: list[Path] = []
    for key in ("cli", "fullsource", "msi", "nsis"):
        destination = output / names[key]
        shutil.copy2(sources[key], destination)
        staged.append(destination)

    fullsource_checksum = output / f"{names['fullsource']}.sha256"
    fullsource_checksum.write_text(
        f"{digest(output / names['fullsource'])}  {names['fullsource']}\n",
        encoding="ascii",
        newline="\n",
    )
    staged.append(fullsource_checksum)

    checksum_lines = [f"{digest(path)}  {path.name}" for path in sorted(staged)]
    checksum_manifest = output / "SHA256SUMS.txt"
    checksum_manifest.write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="ascii",
        newline="\n",
    )
    for path in sorted(staged + [checksum_manifest]):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

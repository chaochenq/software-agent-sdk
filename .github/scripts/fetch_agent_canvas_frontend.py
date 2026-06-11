#!/usr/bin/env python3
"""Fetch the prebuilt agent-canvas frontend into openhands-agent-server."""

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = (
    REPO_ROOT / "openhands-agent-server" / "openhands" / "agent_server" / "frontend"
)
DEFAULT_PACKAGE = "@openhands/agent-canvas"
DEFAULT_VERSION = "1.0.0-rc.7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the published agent-canvas build for packaging."
    )
    parser.add_argument("--package", default=DEFAULT_PACKAGE)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_spec = f"{args.package}@{args.version}" if args.version else args.package
    target = args.target.resolve()

    with tempfile.TemporaryDirectory(prefix="agent-canvas-frontend-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        subprocess.run(
            ["npm", "pack", package_spec, "--pack-destination", str(tmp_path)],
            check=True,
        )
        tarballs = list(tmp_path.glob("*.tgz"))
        if len(tarballs) != 1:
            raise RuntimeError(f"Expected one npm tarball, found {len(tarballs)}")

        with tarfile.open(tarballs[0]) as tar:
            tar.extractall(tmp_path, filter="data")

        build_dir = tmp_path / "package" / "build"
        index_path = build_dir / "index.html"
        if not index_path.is_file():
            raise RuntimeError(f"agent-canvas build missing index.html: {index_path}")

        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(build_dir, target)

    print(f"Fetched {package_spec} frontend into {target}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = REPO_ROOT / "docs" / "source"
DOCS_BUILD = REPO_ROOT / "docs" / "build" / "html"
DEFAULT_WEBSITE = REPO_ROOT.parent / "cancerdynamics-website"
PUBLISHED_SUBDIR = Path("public") / "docs" / "celltraj2"
IGNORED_BUILD_NAMES = {".doctrees", "_modules", "_sources"}


def _remove_readonly(function, path: str, _exc: BaseException) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    function(path)


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, onexc=_remove_readonly)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build/copy celltraj2 Sphinx docs into the cancerdynamics.org "
            "Astro website repo."
        )
    )
    parser.add_argument(
        "--website",
        type=Path,
        default=DEFAULT_WEBSITE,
        help="Path to the cancerdynamics-website repo. Defaults to ../cancerdynamics-website.",
    )
    parser.add_argument("--build", action="store_true", help="Build Sphinx docs before copying.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat Sphinx warnings as errors when used with --build.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without changing the website repo.",
    )
    return parser.parse_args()


def _run_sphinx(*, strict: bool) -> None:
    if DOCS_BUILD.exists():
        _rmtree(DOCS_BUILD)
    cmd = [sys.executable, "-m", "sphinx", "-b", "html", "-E", "-a"]
    if strict:
        cmd.extend(["-W", "--keep-going"])
    cmd.extend([str(DOCS_SOURCE), str(DOCS_BUILD)])
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _validate_website_repo(website: Path) -> Path:
    website = website.resolve()
    if not (website / "package.json").exists() or not (website / "public").exists():
        raise SystemExit(
            f"{website} does not look like the cancerdynamics-website repo "
            "(expected package.json and public/)."
        )
    return website


def _validate_build() -> None:
    index = DOCS_BUILD / "index.html"
    if not index.exists():
        raise SystemExit(
            f"No built docs found at {index}. Run `bash docs/make_docs.sh` "
            "or pass `--build`."
        )


def _copy_docs(website: Path, *, dry_run: bool) -> Path:
    target = (website / PUBLISHED_SUBDIR).resolve()
    allowed_root = (website / "public").resolve()
    if allowed_root not in target.parents:
        raise SystemExit(f"Refusing to publish outside website public/: {target}")

    if dry_run:
        print(f"Would replace {target} with {DOCS_BUILD}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _rmtree(target)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in IGNORED_BUILD_NAMES}

    shutil.copytree(DOCS_BUILD, target, ignore=ignore)
    return target


def main() -> int:
    args = _parse_args()
    website = _validate_website_repo(args.website)
    if args.build:
        _run_sphinx(strict=args.strict)
    _validate_build()
    target = _copy_docs(website, dry_run=args.dry_run)

    if args.dry_run:
        return 0

    print(f"Copied celltraj2 docs to {target}")
    print("Website URL after the Astro site is deployed:")
    print("  https://cancerdynamics.org/docs/celltraj2/")
    print("Next step: review, commit, and push the cancerdynamics-website repo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

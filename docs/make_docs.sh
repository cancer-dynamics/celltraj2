#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/source"
BUILD_DIR="${SCRIPT_DIR}/build/html"

cd "${REPO_ROOT}"

if ! python -c "import sphinx, myst_parser, sphinx_rtd_theme" >/dev/null 2>&1; then
  echo "Missing documentation dependencies." >&2
  echo "Install them with: python -m pip install -r docs/requirements.txt" >&2
  exit 1
fi

SPHINXOPTS=()
if [[ "${STRICT_DOCS:-0}" == "1" ]]; then
  SPHINXOPTS+=("-W" "--keep-going")
fi

rm -rf "${BUILD_DIR}"
python -m sphinx -b html -E -a "${SPHINXOPTS[@]}" "${SOURCE_DIR}" "${BUILD_DIR}"
echo "Built celltraj2 docs at ${BUILD_DIR}/index.html"
echo "To copy them into cancerdynamics.org, run:"
echo "  python docs/publish_to_cancerdynamics.py --build"

#!/usr/bin/env bash
# An isolated, loopback-only server. Pass extra Streamlit options after this script.
set -euo pipefail
studio_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
studio_python="${RESULT_STUDIO_PYTHON:-$studio_repo/.venv-result-studio/bin/python}"
if [ ! -x "$studio_python" ]; then
  echo "Result Studio environment missing; see docs/result_studio.md" >&2
  exit 1
fi
cd "$studio_repo"
exec "$studio_python" -m streamlit run scripts/result_studio.py \
  --server.address 127.0.0.1 --server.port "${RESULT_STUDIO_PORT:-8501}" \
  --server.headless true --server.fileWatcherType none \
  --browser.gatherUsageStats false \
  --theme.base light --theme.primaryColor '#2457DB' \
  --theme.backgroundColor '#F6F8FC' --theme.secondaryBackgroundColor '#EDF1F8' \
  --theme.textColor '#14243D' "$@"

#!/usr/bin/env bash
# run_tests.sh — convenience wrapper around pytest for DSPy contributors.
#
# Usage:
#   ./scripts/run_tests.sh                    # run the full test suite
#   ./scripts/run_tests.sh -k test_signature  # filter by test name
#   ./scripts/run_tests.sh tests/predict/     # run a specific sub-directory
#   ./scripts/run_tests.sh --lf               # re-run only last-failed tests
#
# Any additional arguments are forwarded directly to pytest.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "Running DSPy tests from: ${REPO_ROOT}"
echo "pytest args: -x --tb=short $*"
echo "---"

python -m pytest tests/ -x --tb=short "$@"

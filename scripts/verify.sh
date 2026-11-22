#!/usr/bin/env bash
# One-command verification for a fresh clone.
#
#   bash scripts/verify.sh              # deps -> weights -> tests -> reference vs official -> manifest
#   bash scripts/verify.sh --fast       # skip the (minutes-long) official-forward check
#   bash scripts/verify.sh --no-model   # data/manifest checks only, no checkpoints needed
#
# Exit status is non-zero if any stage fails.  Missing checkpoints are reported
# as SKIP (with the fetch command) rather than as failures, so a clone without
# the 7 GB of third-party weights still gets a useful verdict.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$ROOT/experiments/cm-transformer"
PY="${PYTHON:-python3}"
FAST=0
NO_MODEL=0
for arg in "$@"; do
  case "$arg" in
    --fast) FAST=1 ;;
    --no-model) NO_MODEL=1 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

FAILED=0
header() { printf '\n=== %s\n' "$1"; }
ok()     { printf '  PASS  %s\n' "$1"; }
bad()    { printf '  FAIL  %s\n' "$1"; FAILED=1; }
skip()   { printf '  SKIP  %s\n' "$1"; }

header "1/5 environment"
"$PY" - <<'PY' || FAILED=1
import sys
if sys.version_info < (3, 10):
    sys.exit(f"python {sys.version.split()[0]} is too old; 3.10+ required")
missing = []
for mod in ("numpy", "torch", "pytest"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    sys.exit("missing packages: " + ", ".join(missing) + "  ->  pip install -r requirements.txt")
import numpy, torch
print(f"  PASS  python {sys.version.split()[0]}, numpy {numpy.__version__}, torch {torch.__version__}")
PY

header "2/5 checkpoints"
if [ "$NO_MODEL" = "1" ]; then
  skip "not requested (--no-model)"
elif [ ! -d "$ROOT/models" ]; then
  skip "models/ absent -> python3 scripts/fetch_models.py  (see MODELS.md)"
elif "$PY" "$ROOT/scripts/fetch_models.py" --check; then
  ok "all weight digests match MODELS.md"
else
  bad "checkpoint digests do not match MODELS.md (see the two known source mismatches)"
fi

header "3/5 test suite"
if ( cd "$EXP" && "$PY" -m pytest tests/ -q ); then
  ok "pytest tests/"
else
  bad "pytest tests/"
fi

header "4/5 reference implementation vs official transformers forward"
if [ "$FAST" = "1" ]; then
  skip "not requested (--fast)"
elif ! "$PY" -c "import transformers" >/dev/null 2>&1; then
  skip "transformers not installed (optional: pip install transformers==5.23.0)"
elif [ ! -f "$ROOT/models/Qwen3.5-0.8B/model.safetensors" ]; then
  skip "Qwen3.5-0.8B checkpoint absent"
elif ( cd "$EXP" && "$PY" verify_official.py ); then
  ok "reference matches the official forward pass (see results/verify_official.json)"
else
  bad "verify_official.py"
fi

header "5/5 committed artefacts"
if "$PY" "$ROOT/scripts/check_artefacts.py"; then
  ok "paper2_bundle/DELIVERY_MANIFEST.md digests and result files are intact"
else
  bad "committed artefacts"
fi

if [ "$FAILED" = "0" ]; then
  printf '\nAll stages passed.\n'
else
  printf '\nSome stages failed (see FAIL lines above).\n'
fi
exit "$FAILED"

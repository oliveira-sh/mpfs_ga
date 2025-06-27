#!/usr/bin/env bash

set -euo pipefail

BENCHMARK_DIR="benchmark"
DATASETS_DIR="$BENCHMARK_DIR/datasets"
PREPROCESS="src/discretization.py"
SCRIPT="src/main.py"
REPEATS=1
OUTFILE="$BENCHMARK_DIR/all_discretized_runs_output.txt"

# Flags
DO_DOWNLOAD=false
DO_DISCRETIZE=false
DO_RUN=false

print_usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  -d, --download       Download raw .arff datasets
  -x, --discretize     Discretize all .arff files using $PREPROCESS
  -r, --run            Run the algorithm on *_discretized.arff
  -a, --all            Do download, discretize, and run
  -h, --help           Show this help message
EOF
  exit 1
}

# Parse flags
if [ $# -eq 0 ]; then
  print_usage
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--download)
      DO_DOWNLOAD=true ;;
    -x|--discretize)
      DO_DISCRETIZE=true ;;
    -r|--run)
      DO_RUN=true ;;
    -a|--all)
      DO_DOWNLOAD=true
      DO_DISCRETIZE=true
      DO_RUN=true ;;
    -h|--help)
      print_usage ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage ;;
  esac
  shift
done

if $DO_DOWNLOAD; then
  echo "→ Downloading datasets..."
  mkdir -p "$DATASETS_DIR"
  for f in CellCycle Church Derisi Eisen Expr Gasch1 Gasch2 Sequence SPO; do
    wget -q -O "$DATASETS_DIR/${f}.arff" \
      "https://oliveira-sh.github.io/datasets/${f}.arff"
    echo "  • $f.arff"
  done
  echo "Download complete."
fi

if $DO_DISCRETIZE; then
  echo "→ Discretizing .arff files..."
  mkdir -p "$DATASETS_DIR"
  for raw in "$DATASETS_DIR"/*.arff; do
    if [[ "$raw" == *_discretized.arff ]]; then
      continue
    fi
    echo "  • Processing $(basename "$raw")"
    python3 "$PREPROCESS" "$raw"
  done
  echo "Discretization complete."
fi

if $DO_RUN; then
  echo "→ Running algorithm on discretized datasets..."
  : > "$OUTFILE"
  for trainfile in "$DATASETS_DIR"/*_discretized.arff; do
    base="$(basename "$trainfile" .arff)"
    echo "Dataset: $base" >> "$OUTFILE"
    for i in $(seq 1 $REPEATS); do
      echo "  Run #$i for $base"
      ./pypy/bin/pypy3.11 "$SCRIPT" --train "$trainfile" >> "$OUTFILE" 2>&1
    done
  done
  echo "All runs complete. Combined output in '$OUTFILE'."
fi

if ! $DO_DOWNLOAD && ! $DO_DISCRETIZE && ! $DO_RUN; then
  echo "No action requested."
  print_usage
fi

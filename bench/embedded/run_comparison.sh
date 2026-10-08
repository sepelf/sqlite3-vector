#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 DATASET_DIR WORK_DIR [CPUSET]" >&2
  exit 2
fi

dataset_dir=$(realpath "$1")
work_dir=$(realpath -m "$2")
cpu_set=${3:-${EMBEDDED_CPUSET:-0-15}}
script_dir=$(cd "$(dirname "$0")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)
venv_dir="$work_dir/venv"
build_dir="$work_dir/build"
state_dir="$work_dir/state"
result_dir="$work_dir/results"
report_name="sift1m-embedded-hnsw-comparison.md"
report_path="$result_dir/$report_name"
repo_report="$repo_dir/bench/results/$report_name"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  set +e
  rm -rf \
    "$venv_dir" \
    "$build_dir" \
    "$state_dir" \
    "$work_dir/pip-cache" \
    "$work_dir/sqlite-hnsw.db" \
    "$work_dir/sqlite-hnsw.db-wal" \
    "$work_dir/sqlite-hnsw.db-shm" \
    "$work_dir/chroma" \
    "$work_dir/milvus-lite.db" \
    "$work_dir/capability"
  find "$work_dir" -maxdepth 1 -name 'milvus-lite.db*' -exec rm -rf {} +
  if [[ ! -f "$report_path" ]]; then
    rm -rf "$result_dir"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$work_dir" "$state_dir" "$result_dir"
for file in sift_base.fvecs sift_query.fvecs sift_groundtruth.ivecs; do
  if [[ ! -f "$dataset_dir/$file" ]]; then
    echo "missing SIFT1M file: $dataset_dir/$file" >&2
    exit 1
  fi
done

sqlite_cli=$(command -v sqlite3)
sqlite_version=$($sqlite_cli --version | awk '{print $1}')
if [[ "$sqlite_version" != 3.51.2 ]]; then
  echo "sqlite3 3.51.2 is required; found $sqlite_version at $sqlite_cli" >&2
  exit 1
fi
sqlite_prefix=${SQLITE_PREFIX:-$(dirname "$(dirname "$sqlite_cli")")}
sqlite_library=""
for candidate in "$sqlite_prefix/lib/libsqlite3.so" "$sqlite_prefix/lib64/libsqlite3.so"; do
  if [[ -f "$candidate" ]]; then
    sqlite_library=$candidate
    break
  fi
done
if [[ -z "$sqlite_library" || ! -f "$sqlite_prefix/include/sqlite3.h" ]]; then
  echo "cannot find SQLite headers/library below $sqlite_prefix" >&2
  exit 1
fi
taskset -c "$cpu_set" true

python3 -m venv "$venv_dir"
PIP_CACHE_DIR=${PIP_CACHE_DIR:-$work_dir/pip-cache} \
  "$venv_dir/bin/pip" install \
  --index-url https://pypi.org/simple -r "$script_dir/requirements.lock"

cmake -S "$repo_dir" -B "$build_dir" \
  -DCMAKE_PREFIX_PATH="$sqlite_prefix" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build "$build_dir" --parallel 16
ctest --test-dir "$build_dir" --output-on-failure

export LD_PRELOAD=$sqlite_library
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
python_base=(taskset -c "$cpu_set" "$venv_dir/bin/python")
compare=("${python_base[@]}" "$script_dir/compare.py")
monitor=("${python_base[@]}" "$script_dir/monitor_process.py")
common=(--dataset "$dataset_dir" --work "$work_dir" \
  --extension "$build_dir/sqlite_hnsw.so")

"${compare[@]}" validate \
  --dataset "$dataset_dir" \
  --work "$work_dir" \
  --sqlite-cli "$sqlite_cli" \
  --sqlite-prefix "$sqlite_prefix" \
  --sqlite-version 3.51.2 \
  --output "$state_dir/metadata.json"

for engine in sqlite-hnsw chroma milvus-lite; do
  "${monitor[@]}" --output "$state_dir/${engine}-prepare-memory.json" -- \
    "${compare[@]}" prepare "$engine" "${common[@]}" \
    --output "$state_dir/${engine}-prepare.json"
done
"${compare[@]}" check-sqlite \
  --work "$work_dir" \
  --extension "$build_dir/sqlite_hnsw.so"

: >"$state_dir/queries.jsonl"
for engine in sqlite-hnsw chroma milvus-lite; do
  "${monitor[@]}" --output "$state_dir/${engine}-query-memory.json" -- \
    "${compare[@]}" run "$engine" "${common[@]}" \
    --output "$state_dir/queries.jsonl"
done

: >"$state_dir/cold.jsonl"
for engine in sqlite-hnsw chroma milvus-lite; do
  for repeat in 1 2 3; do
    "${monitor[@]}" --output "$state_dir/${engine}-cold-${repeat}-memory.json" -- \
      "${compare[@]}" cold "$engine" "${common[@]}" \
      --ef 100 --repeat "$repeat" --output "$state_dir/cold.jsonl"
  done
done

"${compare[@]}" report \
  --metadata "$state_dir/metadata.json" \
  --results "$state_dir" \
  --output "$report_path" \
  --date "$(date +%F)"
cp "$report_path" "$repo_report"
echo "embedded comparison report: $repo_report"

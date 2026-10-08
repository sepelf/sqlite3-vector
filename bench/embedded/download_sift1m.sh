#!/bin/sh
set -eu

destination=${1:-/tmp/sqlite-hnsw-sift1m}
archive="$destination/sift-128-euclidean.hdf5"
url=https://ann-benchmarks.com/sift-128-euclidean.hdf5
sha256=dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984
python=${PYTHON:-python3}

mkdir -p "$destination"
if ! printf '%s  %s\n' "$sha256" "$archive" | sha256sum --check --status 2>/dev/null; then
  curl --fail --location --continue-at - --output "$archive" "$url"
fi
printf '%s  %s\n' "$sha256" "$archive" | sha256sum --check --status
"$python" "$(dirname "$0")/hdf5_to_fvecs.py" "$archive" "$destination"

printf 'SIFT1M is ready in %s\n' "$destination"

#!/usr/bin/env python3
"""Convert the ann-benchmarks SIFT1M HDF5 file to TEXMEX vector files."""

from pathlib import Path
import sys

import h5py
import numpy as np


def write_records(dataset: h5py.Dataset, path: Path, dtype: str) -> None:
    width = dataset.shape[1]
    record_type = np.dtype([("dimensions", "<i4"), ("values", dtype, (width,))])
    with path.open("wb") as output:
        for start in range(0, dataset.shape[0], 10_000):
            count = min(10_000, dataset.shape[0] - start)
            records = np.empty(count, dtype=record_type)
            records["dimensions"] = width
            records["values"] = dataset[start : start + count]
            records.tofile(output)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: hdf5_to_fvecs.py INPUT.hdf5 OUTPUT_DIR", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    destination.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as data:
        write_records(data["train"], destination / "sift_base.fvecs", "<f4")
        write_records(data["test"], destination / "sift_query.fvecs", "<f4")
        write_records(
            data["neighbors"], destination / "sift_groundtruth.ivecs", "<i4"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import shutil
import time
from pathlib import Path

import pytest
import zarr
import numpy as np

from slurmkit import slurm_function, submit_function, SlurmParams


def has_slurm() -> bool:
    return shutil.which("sbatch") is not None


pytestmark = pytest.mark.skipif(not has_slurm(), reason="SLURM not found.")


def test_simple_slurm(tmp_path: Path) -> None:

    @slurm_function
    def add(a: int, b: int) -> None:
        c = a + b
        print(f"{a} + {b} = {c}")

    output = tmp_path / "output-%j.out"
    params = SlurmParams(output=output, mem="5M")
    submit_function(add(1, 2), params)

    with open(output) as f:
        value = int(f.read()[-1])
        assert value == 3


def test_slurm_dependency(tmp_path: Path) -> None:

    def _create_ones(size: int, out_path: Path) -> None:
        zarr.ones(shape=(size, size), store=zarr.DirectoryStore(out_path))

    @slurm_function
    def _sum_zarr(
        in_path_1: Path,
        in_path_2: Path,
        out_path: Path,
    ) -> None:
        in_arr1 = zarr.open(in_path_1)
        in_arr2 = zarr.open(in_path_2)
        out_arr = zarr.empty_like(in_arr1, store=zarr.DirectoryStore(out_path))
        out_arr[:] = in_arr1 + in_arr2
    
    size = 5
    params = SlurmParams(output=tmp_path / "slurmkit-test-%j.out")

    create_ones = slurm_function(_create_ones)(size=size)

    paths = [tmp_path / f"{i}.zarr" for i in range(2)]
    
    jobs = [
        submit_function(create_ones, params, out_path=p) for p in paths
    ]

    out_path = tmp_path / "out.zarr"
    submit_function(_sum_zarr(*paths, out_path=out_path), params, dependencies=jobs)

    while not out_path.exists():
        # waiting job to be executed
        time.sleep(3)

    arr = zarr.open(out_path)
    assert np.allclose(arr == 2)



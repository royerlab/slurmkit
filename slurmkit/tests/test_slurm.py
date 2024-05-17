import os
import shutil
from pathlib import Path
from typing import Generator, cast

import numpy as np
import pytest
import zarr

from slurmkit import HAS_SLURM, SlurmParams, slurm_function, slurm_wait, submit_function


@pytest.fixture(autouse=True)
def change_to_shared_dir(monkeypatch: pytest.MonkeyPatch) -> Generator:
    """Moves to temporary shared directory"""
    prev_path = Path(".")

    try:
        path = Path(os.environ["MYDATA"])
    except KeyError:
        path = Path("/tmp")

    assert path.exists()
    path = path / "slurmkit_test"

    if path.exists():
        shutil.rmtree(path)
    path.mkdir()

    monkeypatch.chdir(path)

    yield

    monkeypatch.chdir(prev_path)
    shutil.rmtree(path)


def test_slurm_dependency() -> None:
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
        out_arr[:] = in_arr1[:] + in_arr2[:]

    size = 5
    params = SlurmParams(output="slurmkit-test-%j.out")

    create_ones = slurm_function(_create_ones)(size=size)

    paths = [f"{i}.zarr" for i in range(2)]

    jobs = [cast(int, submit_function(create_ones, params, out_path=p)) for p in paths]

    out_path = "out.zarr"
    sum_job = submit_function(
        _sum_zarr(*paths, out_path=out_path), params, dependencies=jobs
    )

    slurm_wait(sum_job)

    arr = zarr.open(out_path)
    assert np.allclose(arr[:], 2)


@pytest.mark.skipif(not HAS_SLURM, reason="SLURM not found.")
def test_simple_slurm() -> None:
    @slurm_function
    def add(a: int, b: int) -> None:
        c = a + b
        print(f"{a} + {b} = {c}")

    output = "output-%j.out"
    params = SlurmParams(output=output, mem="5M", wait=True)
    job = submit_function(add(1, 2), params)

    with open(output.replace("%j", str(job))) as f:
        assert int(f.read()[-2]) == 3


@pytest.mark.skipif(not HAS_SLURM, reason="SLURM not found.")
def test_int_indexing() -> None:
    @slurm_function
    def _show_int(value: int) -> None:
        assert isinstance(value, int)
        print(f"Value {size}", value)

    output = "slurmkit-test-%j.out"
    params = SlurmParams(output=output)
    size = 10

    jobs = [submit_function(_show_int(), params, value=i) for i in range(size)]

    slurm_wait(jobs)

    for i in range(size):
        path = output.replace("%j", str(jobs[i]))
        with open(path) as f:
            content = f.read()
            assert int(content[-2]) == i

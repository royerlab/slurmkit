# slurmkit: "Minimal" Slurm Toolkit

A python SLURM toolkit.

## Installation

It can be installed using:

```sh
pip install git+https://github.com/royerlab/slurmkit
```

## Examples

### Minimal example

Simple addition executed with a SLURM job.

```python3
from slurmkit import SlurmParams, slurm_function, submit_function

shared_path = Path(...)  # update this

@slurm_function
def add(a: int, b: int) -> None:
    c = a + b
    print(f"{a} + {b} = {c}")

output = shared_path / "output-%j.out"
params = SlurmParams(output=output, mem="5M")
job = submit_function(add(1, 2), params)
```

### Dependencies example

Example with job dependencies and using the `zarr` package to share data between different jobs.

```python3
import zarr

from slurmkit import SlurmParams, slurm_function, submit_function

shared_path = Path(...)  # update this

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
params = SlurmParams(output=shared_path / "slurmkit-test-%j.out")

create_ones = slurm_function(_create_ones)(size=size)

paths = [shared_path / f"{i}.zarr" for i in range(2)]

jobs = [submit_function(create_ones, params, out_path=p) for p in paths]

out_path = shared_path / "out.zarr"
submit_function(_sum_zarr(*paths, out_path=out_path), params, dependencies=jobs)
```

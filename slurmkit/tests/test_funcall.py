from subprocess import CompletedProcess
from typing import cast

import pytest

from slurmkit import slurm_function, submit_function


@pytest.fixture(autouse=True)
def change_test_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path_factory.mktemp("data")
    monkeypatch.chdir(path)


def test_submission_and_funcall() -> None:

    msg = "Sample of length {} from population of length {}."

    @slurm_function
    def _sample(n_pop: int, n_samples: int) -> None:
        import random

        population = list(range(n_pop))
        samples = random.sample(population, k=n_samples)
        print(msg.format(len(population), len(samples)))

    # testing without kwargs
    args = (10, 2)
    args_proc = cast(CompletedProcess, submit_function(_sample(*args), no_sbatch=True))
    assert args_proc.returncode == 0
    assert args_proc.stdout.decode()[:-1] == msg.format(*args)

    # testing with kwargs
    kwargs_proc = cast(
        CompletedProcess,
        submit_function(_sample(n_samples=args[1]), no_sbatch=True, n_pop=args[0]),
    )
    assert kwargs_proc.returncode == 0
    assert kwargs_proc.stdout.decode()[:-1] == msg.format(*args)

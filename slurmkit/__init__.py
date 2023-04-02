import base64
import datetime
import functools
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import cloudpickle
from pydantic import BaseModel, validator

Integers = Union[int, Sequence[int]]


class SlurmParams(BaseModel):
    output: Optional[str] = None
    error: Optional[str] = None
    nodes: Optional[int] = None
    cpus_per_task: Optional[int] = None
    cpus_per_gpu: Optional[int] = None
    mem: Optional[str] = None
    mem_per_cpu: Optional[str] = None
    mem_per_gpu: Optional[str] = None
    partition: Optional[Literal["cpu", "gpu"]] = None
    time: Optional[datetime.timedelta] = None
    kill_on_invalid_dep: Optional[Literal["yes", "no"]] = None
    chdir: Optional[Path] = None
    export: Optional[str] = None
    no_requeue: bool = False
    reservation: Optional[str] = None
    uid: Optional[str] = None
    ntasks: Optional[int] = None
    ntasks_per_core: Optional[int] = None
    ntasks_per_gpu: Optional[int] = None
    ntasks_per_node: Optional[int] = None
    ntasks_per_socket: Optional[int] = None
    wait_all_nodes: Optional[Literal[0, 1]] = None
    exclude: Optional[Union[str, List[str]]] = None
    nodelist: Optional[str] = None
    gres: Optional[Union[str, List[str]]] = None
    gid: Optional[str] = None
    sockets_per_node: Optional[int] = None
    cores_per_socket: Optional[int] = None
    threads_per_core: Optional[int] = None
    threads_per_node: Optional[int] = None

    def check_mutual_exclusivity(
        self, values: Dict[str, Any], exclusive_fields: List[str]
    ) -> None:
        exclusive_values = map(values.get, exclusive_fields)
        num_set_values = sum(map(lambda x: x is not None, exclusive_values))
        if num_set_values > 0:
            raise ValueError(
                f"Mutually exclusive fields `{exclusive_fields}` where set."
            )

    @validator("cpus_per_task")
    def validate_cpus_exclusivity(
        cls, value: Optional[str], values: Dict[str, Any]
    ) -> Optional[str]:
        cls.check_mutual_exclusivity(values, ["cpu_per_task", "cpus_per_gpu"])
        return value

    @validator("mem")
    def validate_mem_exclusivity(
        cls, value: Optional[str], values: Dict[str, Any]
    ) -> Optional[str]:
        cls.check_mutual_exclusivity(values, ["mem", "mem_per_gpu", "mem_per_cpu"])
        return value

    @validator("ntask")
    def validate_ntask_exclusivity(
        cls, value: Optional[str], values: Dict[str, Any]
    ) -> Optional[str]:
        cls.check_mutual_exclusivity(
            values,
            [
                "ntask",
                "ntask_per_core",
                "ntask_per_gpu",
                "ntask_per_node",
                "ntask_per_socket",
            ],
        )
        return value

    @validator("mem", "mem_per_gpu", "mem_per_cpu")
    def validate_mem_type(cls, value: Optional[str], field: str) -> Optional[str]:
        if value is None:
            return None

        mem_types = ("K", "M", "G", "T", "P")
        if not value.upper().endswith(mem_types):
            raise ValueError(
                f"`{field}` must end with on of {mem_types}. Received {value}."
            )

        return value

    def __str__(self) -> str:
        return " ".join(self.tolist())

    def tolist(self) -> List[str]:
        options = []
        for param, value in self:
            if value is None:
                continue

            param = param.replace("_", "-")
            if isinstance(value, (list, tuple)):
                value = ",".join(map(str, value))

            elif isinstance(value, (datetime.timedelta)):
                assert value.microseconds == 0
                value = str(value).replace(",", "-")

            if not isinstance(value, bool):
                options.append(f"--{param}={value}")

            elif value:
                options.append(f"--{param}")

        return options


class SlurmDependencies(BaseModel):
    values: Optional[Integers] = None
    dep_type: Literal[
        "after", "afterany", "afternotok", "afterok", "singleton"
    ] = "afterok"
    minutes: int = 0

    def __str__(self) -> str:
        if self.values is None:
            return ""

        if isinstance(self.values, int):
            jobs = str(self.values)
        else:
            jobs = ",".join(map(str, self.values))

        dependency = f"--dependency={self.dep_type}:{jobs}"

        if self.minutes > 0:
            dependency += f"+{self.minutes}"

        return dependency

    def tolist(self) -> List[str]:
        dependency = str(self)
        if not dependency:
            return []
        return [dependency]


def submit_cli(
    args: Sequence[str],
    slurm_params: Optional[SlurmParams] = None,
    dependencies: SlurmDependencies | Optional[Integers] = None,
) -> int:
    # TODO

    if slurm_params is None:
        slurm_params = SlurmParams()

    if not isinstance(dependencies, SlurmDependencies):
        dependencies = SlurmDependencies(values=dependencies)

    command = (
        ["sbatch", "--parsable"]
        + slurm_params.tolist()
        + dependencies.tolist()
        + list(args)
    )

    complete_proc = subprocess.run(command)
    return complete_proc.returncode


def get_bytes_file_name(data: bytes) -> str:
    # TODO
    # https://stackoverflow.com/questions/4567089/hash-function-that-produces-short-hashes
    return f"{base64.b64encode(hashlib.sha1(data).digest()).decode('utf-8')}.pickle"


def submit_function(
    slurm_function: bytes,
    slurm_params: Optional[SlurmParams] = None,
    dependencies: SlurmDependencies | Optional[Integers] = None,
    **kwargs,
) -> int:
    # TODO
    function_path = Path(get_bytes_file_name(slurm_function))
    if not function_path.exists():
        with open(function_path, mode="wb") as f:
            f.write(slurm_function)

    command = ["funcall", str(function_path)]
    if len(kwargs) > 0:
        command.append("--params")
        for key, value in kwargs.items():
            command += [key, str(value)]

    return submit_cli(command, slurm_params=slurm_params, dependencies=dependencies)


def slurm_function(func: Callable) -> Callable[[Any], bytes]:
    """Wraps a function storing its parameters upon call.

    Reference: https://github.com/joblib/joblib/blob/master/joblib/parallel.py

    Returns
    -------
        Function that returns a delayed function tuple.
    """

    def pickled_function(*args, **kwargs) -> bytes:
        func_tuple = (func, args, kwargs)
        return cloudpickle.dumps(func_tuple)

    try:
        pickled_function = functools.wraps(func)(pickled_function)
    except AttributeError:
        print(f"`functools.wraps` failed on {func.__name__}")

    return pickled_function

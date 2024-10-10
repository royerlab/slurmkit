import base64
import datetime
import functools
import hashlib
import logging
import platform
import subprocess
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union, cast

import cloudpickle
from pydantic.v1 import BaseModel, ConfigDict, validator

Dependency = Union[int, subprocess.CompletedProcess]
Dependencies = Union[Dependency, Sequence[Dependency], None]


LOG = logging.getLogger(__name__)

HAS_SLURM = False
if platform.system() == "Linux":
    HAS_SLURM = subprocess.call(["which", "sbatch"], stdout=subprocess.DEVNULL) == 0


class SlurmParams(BaseModel):
    output: Optional[Union[str, Path]] = None
    error: Optional[Union[str, Path]] = None
    nodes: Optional[int] = None
    gpus: Optional[int] = None
    gpus_per_node: Optional[int] = None
    cpus_per_task: Optional[int] = None
    cpus_per_gpu: Optional[int] = None
    mem: Optional[str] = None
    mem_per_cpu: Optional[str] = None
    mem_per_gpu: Optional[str] = None
    partition: Optional[Literal["cpu", "gpu", "preempted"]] = None
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
    job_name: Optional[str] = None
    wait: bool = False
    constraint: Optional[str] = None
    qos: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def check_mutual_exclusivity(
        values: Dict[str, Any], exclusive_fields: List[str]
    ) -> None:
        exclusive_values = map(values.get, exclusive_fields)
        num_set_values = sum(map(lambda x: x is not None, exclusive_values))
        if num_set_values > 0:
            raise ValueError(
                f"Mutually exclusive fields `{exclusive_fields}` where set."
            )

    @validator("gpus")
    def validate_gpus_exclusivity(
        cls, value: Optional[str], values: Dict[str, Any]
    ) -> Optional[str]:
        cls.check_mutual_exclusivity(values, ["gpus", "gpus_per_node"])
        return value

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

    @validator("ntasks")
    def validate_ntask_exclusivity(
        cls, value: Optional[str], values: Dict[str, Any]
    ) -> Optional[str]:
        cls.check_mutual_exclusivity(
            values,
            [
                "ntasks",
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

    @validator("output")
    def validate_output_directory(
        cls, value: Optional[Union[Path, str]]
    ) -> Optional[Path]:
        if value is None:
            return None

        value = Path(value)
        value.parent.mkdir(exist_ok=True)
        return value

    def __str__(self) -> str:
        return " ".join(self.tolist())

    def tolist(self) -> List[str]:
        options = []
        for param, value in self:
            if value is None or param == "model_config":
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
    values: Optional[List[int]] = None
    dep_type: Literal[
        "after", "afterany", "afternotok", "afterok", "singleton"
    ] = "afterok"
    minutes: int = 0

    @validator("values", pre=True, always=True)
    def validate_values(cls, value: Dependencies) -> List[int]:
        if value is None:
            return []

        if not isinstance(value, list):
            value = [cast(Dependency, value)]

        int_list = [v for v in value if isinstance(v, int)]
        return int_list

    def __str__(self) -> str:
        # is none or empty list
        if self.values is None or (
            not isinstance(self.values, int) and len(self.values) == 0
        ):
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
    dependencies: SlurmDependencies | Dependencies = None,
    no_sbatch: bool = False,
) -> Union[int, subprocess.CompletedProcess]:
    """SLURM sbatch submission using CLI instructions.

    Parameters
    ----------
    args : Sequence[str]
        List of commands and parameters, just like `subprocess.run`.
    slurm_params : Optional[SlurmParams], optional
        sbatch parameters, by default None
    dependencies : SlurmDependencies | Dependencies, optional
        List of dependencies' job IDs, by default None
    no_sbatch : bool, optional
        Executes the command without SLURM, by default False. Used during testing.

    Returns
    -------
    Union[int, subprocess.CompletedProcess]
        Job ID of submitted job or CompletedProcess object when `no_sbatch` is True.
    """

    if slurm_params is None:
        slurm_params = SlurmParams()

    if not isinstance(dependencies, SlurmDependencies):
        dependencies = SlurmDependencies(values=dependencies)

    if not no_sbatch and not HAS_SLURM:
        LOG.warning("Command `sbatch` not found. Running command without SLURM.")
        no_sbatch = True

    if no_sbatch:
        command = list(args)
    else:
        command = (
            ["sbatch", "--parsable"]
            + slurm_params.tolist()
            + dependencies.tolist()
            + list(args)
        )

    str_cmd = " ".join(command)
    LOG.info(f"Executing: {str_cmd}")

    complete_proc = subprocess.run(command, capture_output=True)

    # slurm wait changes return value
    # https://slurm.schedmd.com/sbatch.html#OPT_wait
    if (slurm_params.wait and complete_proc.returncode == 1) or (
        not slurm_params.wait and complete_proc.returncode
    ):
        raise RuntimeError(
            f"Command '{str_cmd}' failed with {complete_proc.stderr.decode()}"
        )

    return complete_proc if no_sbatch else int(complete_proc.stdout)


def get_bytes_file_name(data: bytes) -> str:
    """Creates a unique name for each `data`."""
    name = base64.b64encode(hashlib.sha1(data).digest()).decode()
    name = name.replace("/", "_")
    name = name.replace("=", "_")
    return f"{name}.pickle"


def submit_function(
    slurm_function: tuple[str, bytes],
    slurm_params: Optional[SlurmParams] = None,
    dependencies: SlurmDependencies | Dependencies = None,
    no_sbatch: bool = False,
    **kwargs: Any,
) -> Union[int, subprocess.CompletedProcess]:
    """SLURM sbatch submission from a function.

    Parameters
    ----------
    slurm_function : tuple[str, bytes]
        Function to be processed on SLURM.
    slurm_params : Optional[SlurmParams], optional
        sbatch parameters, by default None
    dependencies : SlurmDependencies | Dependencies, optional
        List of dependencies' job IDs, by default None
    no_sbatch : bool, optional
        Executes the command without SLURM, by default False. Used during testing.
    kwargs : Any
        Additional keyword arguments. These arguments aren't pickled and are provided through the CLI submission.
        Using `kwargs` is the recommended way of using indexing or emulating a SLURM job-array.

    Returns
    -------
    Union[int, subprocess.CompletedProcess]
        Job ID of submitted job or CompletedProcess object when `no_sbatch` is True.
    """

    func_name, func = slurm_function

    func_path = Path(get_bytes_file_name(func)).resolve()
    if not func_path.exists():
        with open(func_path, mode="wb") as f:
            f.write(func)

    command = ["funcall", str(func_path)]
    if len(kwargs) > 0:
        command.append("--params")
        for key, value in kwargs.items():
            command += [key, str(value)]

    if slurm_params is None:
        slurm_params = SlurmParams()

    if slurm_params.job_name is None:
        slurm_params = cast(SlurmParams, slurm_params.copy())
        slurm_params.job_name = func_name

    return submit_cli(
        command,
        slurm_params=slurm_params,
        dependencies=dependencies,
        no_sbatch=no_sbatch,
    )


def slurm_function(func: Callable) -> Callable:
    """Wraps a function storing its parameters upon call.

    Reference: https://github.com/joblib/joblib/blob/master/joblib/parallel.py

    Returns
    -------
        Function that returns a delayed function tuple.
    """

    def pickled_function(*args, **kwargs) -> tuple[str, bytes]:
        func_tuple = (func, args, kwargs)
        LOG.info(
            f"Pickling function `{func.__name__}` with args={args} and kwargs={kwargs}."
        )
        return func.__name__, cloudpickle.dumps(func_tuple)

    try:
        pickled_function = functools.wraps(func)(pickled_function)
    except AttributeError:
        LOG.warn(f"`functools.wraps` failed on {func.__name__}")

    return pickled_function


def slurm_wait(
    dependencies: SlurmDependencies | Dependencies = None,
) -> Optional[Dependency]:
    """
    Creates a dummy job that blocks the execution until all dependencies are finished.

    Parameters
    ----------
    dependencies : SlurmDependencies | Dependencies, optional

    Returns
    -------
    Optional[Dependency]
        Job ID of dummy job or None when `sbatch` is not available.
    """
    if not HAS_SLURM:
        warnings.warn("Command `sbatch` not found. Ignoring `slurm_wait`.")
        return None

    slurm_params = SlurmParams(
        mem="1M",
        wait=True,
        job_name="WAIT_SK",
        output="/tmp/slurmkit_wait_output_%j.out",
    )

    return submit_cli(
        ['--wrap="date"'],
        slurm_params=slurm_params,
        dependencies=dependencies,
    )

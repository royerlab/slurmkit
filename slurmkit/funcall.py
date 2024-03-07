import argparse
import logging
import pickle
import sys
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, get_type_hints

LOG = logging.getLogger(__name__)


def _type_cast(types: Dict[str, Callable], param_name: str, param: str) -> Any:
    """
    Casts a parameter to its type annotation.

    TODO: Add test that covers the case where the type annotation is not found.

    Parameters
    ----------
    types : Dict[str, Callable]
        Dictionary of type annotations.
    param_name : str
        Name of the parameter.
    param : str
        Value of the parameter.

    Returns
    -------
    Any
        The parameter casted to its type annotation.
    """

    try:
        var_type = types[param_name]

        if var_type == bool:
            # NOTE: boolean typecast bool("False") == True
            is_false = param.lower() == "false" or param == "0"
            return not is_false

        return var_type(param)

    except KeyError:
        try:
            LOG.warning(
                f"Could not find type for parameter {param_name}. Using `float` as default."
            )
            return float(param)

        except ValueError:
            LOG.warning(
                f"Could not cast parameter {param_name} to `float`. Using `str` as default."
            )
            return str(param)


def _parse_args(argv: Optional[Sequence[str]]) -> Tuple[Callable, Sequence, Dict]:
    parser = argparse.ArgumentParser(
        "funcall",
        description="Executes a pickled (function, args, kwargs) tuple given its path.",
    )
    parser.add_argument(
        "FUNC-TUPLE-PATH",
        type=str,
        help="Path to pickled object of a tuple of (function, args and kwargs).",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        metavar=("name", "value"),
        required=False,
        default=tuple(),
        help="Keyword parameter name and value pairs.",
    )

    cli_args = vars(parser.parse_args(argv))

    with open(cli_args["FUNC-TUPLE-PATH"], mode="rb") as f:
        func, args, kwargs = pickle.load(f)

    types = get_type_hints(func)

    params = {
        name: _type_cast(types, name, value)
        for name, value in zip(cli_args["params"][::2], cli_args["params"][1::2])
    }

    for key in params:
        if key in kwargs:
            raise ValueError(
                f"Found duplicated key {key} in `--params` and pickled function kwargs."
            )

    kwargs.update(params)

    return func, args, kwargs


def main(argv: Optional[Sequence[str]] = None) -> None:
    func, args, kwargs = _parse_args(argv)
    func(*args, **kwargs)


if __name__ == "__main__":
    main(sys.argv[1:])

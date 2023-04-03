import argparse
import pickle
import sys
from typing import Callable, Dict, Optional, Sequence, Tuple, get_type_hints


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
        name: types[name](value)
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

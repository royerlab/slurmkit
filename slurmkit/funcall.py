import argparse
import pickle
from typing import Callable, Dict, Sequence, Tuple, get_type_hints


def _parse() -> Tuple[Callable, Sequence, Dict]:
    parser = argparse.ArgumentParser(
        "funcall",
        description="Executes a function given a pickled function, args, kwargs tuple path",
    )
    parser.add_argument(
        "func-tuple-path",
        required=True,
        type=str,
        help="Path to pickled object of a tuple of function, args and kwargs.",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        metavar=("name", "value"),
        required=False,
        help="Keyword parameter name and value pairs",
    )

    cli_args = parser.parse_args()

    with open(cli_args.func_tuple_path, mode="rb") as f:
        func, args, kwargs = pickle.load(f)

    types = get_type_hints(func)

    params = {name: types[name](value) for name, value in cli_args.params}

    for key in params:
        if key in kwargs:
            raise ValueError(
                f"Found duplicated key {key} in `--params` and pickled function kwargs."
            )

    kwargs.update(params)

    return func, args, kwargs


def main() -> None:
    func, args, kwargs = _parse()
    func(args, kwargs)


if __name__ == "__main__":
    main()

from typing import Any


def set(a: Any, b: Any) -> Any:
    if a is not None:
        return a
    elif b is not None:
        return b
    raise ValueError(
        f"Only one of 'a': {a} or 'b': {b} can be `None`")
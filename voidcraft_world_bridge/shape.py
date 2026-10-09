"""One reader for untrusted JSON — a server reply, a browser body — where a
missing or mistyped field is a normal case, never an error.

`typed(value, kind, default)` replaces the `x if isinstance(x, kind) else default`
idiom that would otherwise repeat at every field read. mypy narrows through it,
which it cannot do through the inline form when `x` is itself a `.get()`.
"""
from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def typed(value: object, kind: type[T], default: T) -> T:
    """`value` when it is a `kind`, else `default`."""
    return value if isinstance(value, kind) else default

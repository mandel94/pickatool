from collections.abc import Iterator
from typing import NamedTuple, Optional


class FCluster(NamedTuple):
    """Feature cluster class."""

    name: str
    nodes: set = {}
    pairs: Optional[list] = []

    def __iter__(self) -> Iterator:
        # Return an iterator with the object itself
        return iter([self])

from typing import NamedTuple, Optional


class FCluster(NamedTuple):
    """Feature cluster class."""
    name: str
    nodes: set = {}
    pairs: Optional[list] = []
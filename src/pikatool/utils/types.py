from typing import (
    Literal,
    Union,
    Any,
    Callable,
    Optional,
    TypeAlias,
    Iterable,
    NamedTuple,
)
from dateutil import parser
from enum import Enum
import pandas as pd
import numpy as np
from .datetime_parser import ItalianParserInfo
from dataclasses import dataclass, Field


# NETWORK ANALYSIS ----------------------------------------------------------

Node: TypeAlias = tuple[int, str]

Edge: TypeAlias = tuple[int, int, str, float]


# STRIGN HANDLING -----------------------------------------------------------


# PARSING -------------------------------------------------------------------

AvailableLanguages: TypeAlias = Literal["italian"]


class ParsingInfoByLanguage(Enum):
    ITALIAN: parser.parserinfo = ItalianParserInfo


# DATA HANDLING -------------------------------------------------------------

AvailableEngines: TypeAlias = Literal["pandas"]


DataType: TypeAlias = Union[np.ndarray, Iterable, dict, pd.DataFrame]


class DataFrame(Enum):
    PANDAS = pd.DataFrame


class SquareDataframe(pd.DataFrame):
    # Validate if the dataframe is square
    def __init__(self) -> None:
        super().__init__()
        self._validate()

    def _validate(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError("Dataframe is not square")
        return self


# FEATURE CLUSTERING ----------------------------------------------------------------


class FCluster(NamedTuple):
    """Feature cluster class."""

    name: str
    nodes: set = {}
    pairs: Optional[list] = []

from typing import Literal
from dateutil import parser
from enum import Enum
import pandas as pd
from .datetime_parser import ItalianParserInfo


AvailableLanguages = Literal["italian"]

AvailableEngines = Literal["pandas"]

# ---------------------------------------------------------------------------
# Network Analysis

Node = tuple[int, str]

Edge = tuple[int, int, str, float]

# ---------------------------------------------------------------------------


class ParsingInfoByLanguage(Enum):
    ITALIAN: parser.parserinfo = ItalianParserInfo


# ---------------------------------------------------------------------------


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
    

from abc import ABC, abstractmethod
from typing import Literal
import pandas as pd
from typing import Callable
from ..utils.types import DataFrame

AvailableLoadFrameworks = Literal["pandas", "polar"]

AvailableFileTypes = Literal["csv", "json", "xlsx", "parquet"]


class DataLoader(ABC):

    LOADERS: dict[AvailableLoadFrameworks, dict[AvailableFileTypes, Callable]] = {
        "pandas": {
            "csv": pd.read_csv,
            "json": pd.read_json,
            "parquet": pd.read_parquet,
            "xlsx": pd.read_excel,
        }
    }

    def __init__(self, path: str):
        if self._get_extension(path) in AvailableFileTypes.__args__:
            self.path = path
            self.file_ext = self._get_extension(path)
        else:
            raise ValueError(
                f"File type not supported:\
                             {self._get_extension(path)}"
            )

    @abstractmethod
    def load_data(self):
        pass

    def _get_extension(self, path: str) -> str:
        return path.split(".")[-1]


class PandasDataLoader(DataLoader):

    PANDAS_LOADER = DataLoader.LOADERS["pandas"]

    def __init__(self, path: str):
        super().__init__(path)

    def load_data(self, **kwargs) -> DataFrame:
        loader_function = self.PANDAS_LOADER[self.file_ext]
        return loader_function(self.path, **kwargs)

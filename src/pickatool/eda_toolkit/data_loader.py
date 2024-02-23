from abc import ABC, abstractmethod
from typing import Literal, TypeAlias, Optional
import pandas as pd
import re
import numpy as np
from typing import Callable
from ..utils.types import DataFrame

AvailableLoadFrameworks = Literal["pandas", "polar"]

AvailableFileTypes = Literal["csv", "json", "xlsx", "parquet", "txt"]

AllowedSeparators: TypeAlias = Literal[",", ";", "\t"]


class DataLoader(ABC):

    LOADERS: dict[AvailableLoadFrameworks, dict[AvailableFileTypes, Callable]] = {
        "pandas": {
            "csv": pd.read_csv,
            "json": pd.read_json,
            "parquet": pd.read_parquet,
            "xlsx": pd.read_excel,
            "txt": pd.read_table,
        }
    }

    SAVERS: dict[AvailableLoadFrameworks, dict[AvailableFileTypes, Callable]] = {
        "pandas": {
            "csv": pd.DataFrame.to_csv,
            "json": pd.DataFrame.to_json,
            "parquet": pd.DataFrame.to_parquet,
            "xlsx": pd.DataFrame.to_excel,
        }
    }

    def _get_separator(self) -> str:
        candidate_separators: tuple[str] = AllowedSeparators.__args__
        n_of_lines = 100
        with open(self.path, "r", encoding=self.encoding) as file:
            ## Read the first 5 lines of the file
            lines = [file.readline() for _ in range(n_of_lines)]
            for separator in candidate_separators:
                seps_in_lines = [
                    [sep for sep in re.findall(separator, line)] for line in lines
                ]
                how_many_seps_each_line = [
                    len(seps_in_line) for seps_in_line in seps_in_lines
                ]
                how_many_seps_in_first_line = how_many_seps_each_line[0]
                # If each line has the same number of separator, than the
                # candidate separator is a good guess. This gets more reasonable
                # as the number of lines tested increases.
                avg_how_how_many_seps_each_line = np.array(
                    how_many_seps_each_line
                ).mean()
                if (
                    avg_how_how_many_seps_each_line
                ):  # When zero, the separator has not been found
                    if avg_how_how_many_seps_each_line == how_many_seps_in_first_line:
                        return separator
        raise ValueError(
            "No separator found in provided file. The following\
                         separators were tested: f{candidate_separators}"
        )

    def __init__(self, path: str, encoding: Optional[str] = "utf-8"):
        if self._get_extension(path) in AvailableFileTypes.__args__:
            self.path = path
            self.encoding = encoding
            self.file_ext = self._get_extension(path)
        else:
            raise ValueError(f"File type not supported: {self._get_extension(path)}")

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def save_data(self):
        pass

    def _get_extension(self, path: str) -> str:
        return path.split(".")[-1]


class PandasDataLoader(DataLoader):

    PANDAS_LOADER = DataLoader.LOADERS["pandas"]

    PANDAS_SAVER = DataLoader.SAVERS["pandas"]

    def __init__(self, path: str, encoding: Optional[str] = "utf-8"):
        super().__init__(path, encoding)

    def load_data(self, **kwargs) -> DataFrame:
        loader_function = self.PANDAS_LOADER[self.file_ext]
        if self.file_ext == "txt":
            separator = self._get_separator()
            return loader_function(self.path, sep=separator, **kwargs)
        return loader_function(self.path, **kwargs)

    def save_data(self, data: DataFrame, path: str, **kwargs) -> None:
        saver_function = self.PANDAS_SAVER[self.file_ext]
        saver_function(data, path, **kwargs)

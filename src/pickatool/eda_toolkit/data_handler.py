from abc import ABC, abstractmethod
from typing import Literal, Union
import pandas as pd
from .service import Task
from ..utils.types import ParsingInfoByLanguage, AvailableLanguages
from dateutil import parser
from .data_loader import PandasDataLoader
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import DistanceMetric
from ..graph_toolkit.NetworkAnalysis import NetworkAnalysis

from ..utils.types import DataFrame, SquareDataframe


# DataHandler is the entry point for all data operations
class DataHandler(ABC):
    def __init__(self, path):
        self.path = path
        self.task = Task()

    @abstractmethod
    def add_data(self, name: str, data: DataFrame):
        """Add data to the data handler."""
        pass

    @abstractmethod
    def copy_data(self, copy_name):
        """Create a deep copy of the data hold by data handler."""
        pass

    @staticmethod
    def _get_locale_parser_info(language: AvailableLanguages) -> parser.parserinfo:
        return ParsingInfoByLanguage[language.upper()].value

    @abstractmethod
    def load(self) -> DataFrame:
        pass

    @abstractmethod
    def count_unique_values(self, column: str):
        pass

    @abstractmethod
    def get_number_of_missing_values(self, column: str):
        pass

    @abstractmethod
    def get_missing_values(self, column: str):
        pass

    @abstractmethod
    def where(
        self,
        column: str,
        condition: Literal["gt", "gte", "lt", "lte", "eq"],
        value: Union[int, float, str],
    ):
        pass

    @abstractmethod
    def where_missing(self, column: str):
        pass

    @abstractmethod
    def parse_datetime(self, column: str, language: AvailableLanguages = "english"):
        pass

    # @abstractmethod
    # def get_outliers(self, column: Union[str, list(str)], method: Literal["z-score", "iqr"]):
    #     '''Returns the index list of outliers in the given column(s) of the dataset.'''
    #     pass

    @abstractmethod
    def _get_outliers_z_score(self, column: str):
        pass

    @abstractmethod
    def _get_outliers_iqr(self, column: str):
        pass


class PandasDataHandler(DataHandler):

    def __init__(self, path: str):
        self.path = path
        self.data = pd.DataFrame()

    @staticmethod
    def compute_distances(
        data: DataFrame,
        axis: Literal["columns", "rows"] = "columns",
        metric: str = "euclidean",
    ) -> DataFrame:
        return NetworkAnalysis.compute_distances(data, axis=axis, metric=metric)
        

    def add_data(self, name: str, data: DataFrame):
        setattr(self, name, data)
        return self

    def copy_data(self, copy_name: str):
        """Create a deep copy of the data hold by data handler."""
        setattr(self, copy_name, self.data.copy(deep=True))
        print("Data copied successfully")
        return self

    def count_unique_values(self, column: str):
        print(f"Count of unique '{column}': {len(self.data[column].value_counts())}")
        return self.data[column].value_counts()

    def get_number_of_missing_values(self, column: str):
        tot_missing = self.data[column].isnull().sum()
        print(f"Number of missing values in '{column}': {tot_missing}")
        return tot_missing

    def get_missing_values(self, column: str):
        return self.data[self.data[column].isnull()]

    def where(
        self,
        column: str,
        condition: Literal["gt", "gte", "lt", "lte", "eq"],
        value: Union[int, float, str],
    ):
        try:
            if condition == "gt":
                return self.data[self.data[column] > value]
            elif condition == "gte":
                return self.data[self.data[column] >= value]
            elif condition == "lt":
                return self.data[self.data[column] < value]
            elif condition == "lte":
                return self.data[self.data[column] <= value]
            elif condition == "eq":
                return self.data[self.data[column] == value]
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def where_missing(self, column: str):
        return self.data[self.data[column].isnull()]

    def parse_datetime(self, column: str, language: AvailableLanguages = "english"):
        parser_info = self._get_locale_parser_info(language)
        return self.data[column].apply(
            lambda x: parser.parse(x, parserinfo=parser_info)
        )

    def load(self, **kwargs) -> DataFrame:
        data_loader = PandasDataLoader(self.path)
        self.data = data_loader.load_data(**kwargs)
        return self

    # def get_outliers(self, column: Union[str, list(str)], method: Literal["z-score", "iqr"]):
    #     # Validate self.data[column]. It must be numeric
    #     # if not is_numeric_dtype(self.data[column]):
    #     #     raise ValueError(f"Column '{column}' is not numeric")
    #     pass

    def _get_outliers_z_score(self, column, threshold: float = 1.5):
        # TODO: Implement Z-Score method
        raise NotImplementedError("Z-Score method not implemented yet")

    def _get_outliers_iqr(self, column, threshold: float = 1.5):
        # TODO: Implement IQR method
        raise NotImplementedError("IQR method not implemented yet")

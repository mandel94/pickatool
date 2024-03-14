from typing import Literal, Union, Any, Callable, Optional, Self, TypeVar
from pandas._typing import IndexLabel
import pandas as pd
import numpy as np
from ..utils.types import ParsingInfoByLanguage, AvailableLanguages
from dateutil import parser
from .data_loader import PandasDataLoader, AvailableFileTypes
from ..graph_toolkit.NetworkAnalysis import NetworkAnalysis
import os
from matplotlib import pyplot as plt


def _path_exists(path: str) -> bool:
    return os.path.exists(path)


def _get_locale_parser_info(language: AvailableLanguages) -> parser.parserinfo:
    return ParsingInfoByLanguage[language.upper()].value


def compute_distances(
    data: pd.DataFrame,
    axis: Literal["columns", "rows"] = "columns",
    metric: str = "euclidean",
) -> pd.DataFrame:
    return NetworkAnalysis.compute_distances(data, axis=axis, metric=metric)


class DataHandler:

    def __init__(
        self, /, path: Optional[str] = None, data: Optional[pd.DataFrame] = None
    ):
        """Initialize the DataHandler class."""
        if path:
            if not _path_exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            self.path = path
        elif data is not None:
            self.data = data
        else:
            raise ValueError(
                "No data provided. Please provide a path or a pd.DataFrame."
            )

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def add_data(self, name: str, data: pd.DataFrame):
        setattr(self, name, data)
        return self

    def update_data(self, data: pd.DataFrame) -> Self:
        self.data = data
        print("Data updated successfully")

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

    def get_unique_values(self, column: str, **kwargs) -> pd.Series:
        return self.data[column].unique(**kwargs)

    def count_unique_values(self, column: str, **kwargs) -> pd.Series:
        return self.data[column].value_counts(**kwargs)

    def plot_unique_values(self, column: str, **kwargs):
        counts = self.data[column].value_counts(**kwargs, sort=True)
        _, ax = plt.subplots()
        ax.bar(counts.index, counts.values)
        ax.set_xlabel(column)
        ax.set_ylabel("Counts")
        ax.set_title(f"Unique values in {column}")
        plt.show()

    def count_missing(self, axis: Literal["rows", "columns"], **kwargs):
        axis = 1 if axis == "rows" else 0
        return self.data.isnull().sum(axis=axis, **kwargs)

    def get_where(
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

    def get_where_equal(self, column: str, value: Any):
        return self.data[self.data[column] == value]

    def get_where_missing(self, column: str):
        return self.data[self.data[column].isnull()]

    def get_where_type(self, column: str, dtype: Union[str, int, float, bool]):
        matching_type = self.data[
            self.data[column].apply(lambda x: isinstance(x, dtype))
        ]
        return matching_type[column]

    def set_where_equal(
        self,
        column: str,
        value: Union[int, float, str],
        new_value: Union[int, float, str],
        return_new_series: bool = False,
        inplace: bool = True,
    ) -> Union[Self, pd.Series]:
        new_series = self.data[column].replace(value, new_value, inplace=inplace)
        return new_series if return_new_series else self

    def set_where_missing(
        self,
        column: str,
        new_value: Union[int, float, str],
        return_new_series: bool = False,
        inplace: bool = True,
    ) -> Union[Self, pd.Series]:
        new_series = self.data[column].fillna(new_value, inplace=inplace)
        return new_series if return_new_series else self

    def set_where_not_missing(
        self,
        column: str,
        new_value: Union[int, float, str],
        return_new_series: bool = False,
        inplace: bool = True,
    ) -> Union[Self, pd.Series]:

        new_series = self.data[column].apply(
            lambda x: new_value if not pd.isnull(x) else x
        )
        if inplace:
            self.data[column] = new_series
        return new_series if return_new_series else self

    def set_where_else(
        self,
        column: str,
        condition: Callable,
        new_value: Union[int, float, str],
        alternative_value: Union[int, float, str],
        return_new_series: bool = False,
        inplace: bool = True,
    ) -> Union[Self, pd.Series]:
        """Set a value in a column where a condition is met, otherwise set an alternative value.

        Args:
            column (str): The column to apply the condition to.
            condition (Callable): A function that returns a boolean value.
            new_value (Union[int, float, str]): The value to set where the condition is met.
            alternative_value (Union[int, float, str]): The value to set where the condition is not met.
            return_new_series (bool, optional): Whether to return the new series. Defaults to False.
            inplace (bool, optional): Whether to update the dataframe inplace. Defaults to True.

        Returns:
            Union[Self, pd.Series]: The updated dataframe or the new series.

        Example:
            data_handler.set_where_else('age', lambda x: x >= 18, Adult, Minor)
        """
        new_series = self.data[column].apply(
            lambda x: new_value if condition(x) else alternative_value
        )
        if inplace:
            self.data[column] = new_series
        return new_series if return_new_series else self

    def set_type(self, column: str, new_type: Union[str, int, float, bool]) -> Self:
        self.data = self.data.astype({column: new_type})
        return self

    def rename_column(self, old_name: str, new_name: str, inplace: bool = True):
        return self.data.rename(columns={old_name: new_name}, inplace=inplace)

    def reindex_column(self, column: str, new_index: int):
        """Reindex a column in the dataframe.
        It pops the column and reinserts it at the new index. It keeps the
        original column name. The dataframe is updated inplace.
        """
        col = self.data.pop(column)
        self.data.insert(new_index, col.name, col)  # Update the dataframe inplace
        return self

    def column_apply(
        self, column: str, function: Callable, inplace=True
    ) -> pd.DataFrame:
        """Apply a function to a column of the dataframe."""
        new_col = self.data[column].apply(function)
        if inplace:
            self.data[column] = new_col
        return new_col

    def column_names_apply(self, function: Callable) -> Any:
        """Apply a function to the column names of the dataframe."""
        return self.data.rename(columns=function, inplace=True)

    def remove_where(
        self, column: str, condition: Callable, inplace: bool = True
    ) -> Self:
        """Remove rows or columns based on a condition.

        Args:
            column (str): The column to apply the condition to.
            condition (Callable): A function that returns a boolean value.

        Returns:
            pd.DataFrame: The dataframe with the rows or columns removed.
        """
        # If condition is not a callable, raise an error
        if not callable(condition):
            raise ValueError("Condition must be a callable function")

        filtered_data = self.data[np.invert(self.data[column].apply(condition))]
        if inplace:
            self.data = filtered_data
        return Self

    def remove_where_index_equal(
        self, value: Union[int, float, str], inplace: bool = True
    ) -> pd.DataFrame:
        """Remove rows based on the index value."""
        filtered_data = self.data.drop(index=value, inplace=inplace)
        return filtered_data

    def remove_columns(self, columns: IndexLabel, inplace: bool = True) -> pd.DataFrame:
        """Remove columns from the dataframe.

        Args:
            columns (Union[list[str] | str]): The columns to remove.
            inplace (bool, optional): Whether to update the dataframe inplace. Defaults to True.
        """
        filtered_data = self.data.drop(columns=columns, inplace=inplace)
        return filtered_data

    def one_hot_encode(self, column: str, inplace: bool = True) -> pd.DataFrame:
        df_with_dummies = pd.get_dummies(self.data[column])
        if inplace:
            self.data = pd.concat([self.data, df_with_dummies], axis=1)
        return df_with_dummies

    def pivot_grouped_categories_count(
        self,
        column: str,
        key: Union[str, list[str]],
        suffix: Optional[str] = "_count",
        **kwargs,
    ) -> pd.Series:
        """Group by the provided key, and count the occurrences of each category in the provided column for each group.

        This command is useful when you want to count the occurrences of each category in a column, grouped by another column.
        For example, you have a column with the time of the day when a user made a purchase,
        and you want to count the occurrences of each time slot for each user (morning, afternoon, evening and night).

        Returns the dataframe with only the pivoted categories,
        indexed by the provided key.

        Args:
            column (str): The column to pivot.
            key (Union[str, list[str]]): The column(s) to group by.4
            suffix (Optional[str], optional): The suffix to add to the new columns. Defaults to "_count".
            **kwargs: Additional arguments to pass to pd.get_dummies.

        Returns:
            pd.DataFrame: The pivoted dataframe.

        Example:
            You have 5 users, each with a number of purchases:
            user_id | purchase_time
            1       | morning
            1       | morning
            1       | afternoon
            2       | morning
            2       | night
            3       | afternoon
            3       | afternoon

            >>> data_handler.pivot_categories_count('purchase_time', 'user_id')
            user_id (index) | morning | afternoon | night
            1               | 2       | 1         | 0
            2               | 1       | 0         | 1
            3               | 0       | 2         | 0

        """
        categories = self.data[column].value_counts().index
        one_hot_encoded_df = self.one_hot_encode(column, inplace=False)
        # concat with key column
        one_hot_encoded_df = pd.concat([self.data[key], one_hot_encoded_df], axis=1)
        args = {f"{cat}{suffix}": (cat, "sum") for cat in categories}
        categories_count_per_group = one_hot_encoded_df.groupby(key).agg(**args)
        return categories_count_per_group

    def parse_datetime(self, column: str, language: AvailableLanguages = "english"):
        parser_info = _get_locale_parser_info(language)
        return self.data[column].apply(
            lambda x: parser.parse(x, parserinfo=parser_info)
        )

    def merge(
        self,
        to_merge: Union["DataHandler", pd.DataFrame],
        print_before_after: bool = False,
        inplace: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Take the data attribute and merge it with another dataframe"""
        if to_merge.__class__.__name__ == "DataHandler":
            to_merge = to_merge.data
        new_data = pd.merge(self.data, to_merge, **kwargs)
        if print_before_after:
            print(
                f"Shape before merge: {self.data.shape}\n"
                f"Shape after merge: {new_data.shape}\n"
            )
            new_columns = [
                col for col in new_data.columns if col not in self.data.columns
            ]
            print(f"New columns after merge: {new_columns}\n")
        if inplace:
            self.data = new_data
        return self.data

    def load(self, **loadKw) -> pd.DataFrame:
        """Load data using path provided in the constructor."""
        if not self.path:
            raise ValueError(
                "No path provided. You should provide a path before loading data."
            )
        data_loader = PandasDataLoader(self.path)
        self.data = data_loader.load_data(**loadKw)
        return self.data

    def save(self, path: str, **kwargs) -> None:
        data_loader = PandasDataLoader(path)
        data_loader.save_data(self.data, path=path, **kwargs)

    def _get_outliers_z_score(self, column, threshold: float = 1.5):
        # TODO: Implement Z-Score method
        raise NotImplementedError("Z-Score method not implemented yet")

    def _get_outliers_iqr(self, column, threshold: float = 1.5):
        # TODO: Implement IQR method
        raise NotImplementedError("IQR method not implemented yet")

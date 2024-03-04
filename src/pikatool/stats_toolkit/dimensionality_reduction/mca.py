from mca import MCA as _MCA
import prince
from typing import Union, TypeAlias, Any, Literal, NamedTuple, Optional, Iterable
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class MCA(ABC):
    def __init__(self, data: pd.DataFrame, **kwargs) -> None:
        self.data = data
        self.obj = self._run_mca(**kwargs)

    @abstractmethod
    def _run_mca(self, **kwargs) -> Any:
        pass
    
    @property
    @abstractmethod
    def eigenvalues(self) -> Any:
        pass

    @property
    @abstractmethod
    def row_coordinates(self) -> Any:
        pass

    @property
    @abstractmethod
    def column_coordinates(self) -> Any:
        pass

    @property
    @abstractmethod
    def row_contributions(self) -> Any:
        pass

    @property
    @abstractmethod
    def column_contributions(self) -> Any:
        pass

    @property
    @abstractmethod
    def row_cosine_similarities(self) -> Any:
        pass

    @property
    @abstractmethod
    def column_cosine_similarities(self) -> Any:
        pass

    @abstractmethod
    def plot_coordinates(self) -> Any:
        pass

    @abstractmethod
    def scree_plot(self) -> Any:
        pass



class MCA_v0(MCA):
    """Run MCA analysis, based on the `mca` package.

    GitHub Repo -> https://github.com/esafak/mca
    """

    def __init__(self, data: pd.DataFrame, **kwargs) -> None:
        super().__init__(data, **kwargs)

    def _run_mca(self, **kwargs) -> Any:
        return _MCA(self.data, **kwargs)

    @property
    def eigenvalues(self) -> Any:
        return self.obj.L

    @property
    def row_coordinates(self) -> Any:
        return self.obj.fs_r

    @property
    def column_coordinates(self) -> Any:
        return self.obj.fs_c
    
    @property
    def row_cosine_similarities(self) -> Any:
        return self.obj.cos_r
    
    @property
    def column_cosine_similarities(self) -> Any:
        return self.obj.cos_c
    
    @property
    def row_contributions(self) -> Any:
        pass

    @property
    def column_contributions(self) -> Any:
        pass
    
    def plot_coordinates(self) -> Any:
        pass

    def scree_plot(self) -> Any:
        pass


class MCA_v1(MCA):
    """Run MCA analysis.

    V1 implements this -> https://vxy10.github.io/2016/06/10/intro-MCA/
    """

    def __init__(self, data: pd.DataFrame, **kwargs) -> None:
        super().__init__(data, **kwargs)

    def _run_mca(self, **kwargs) -> Any:
        pass

    @property
    def eigenvalues(self) -> Any:
        pass

    @property
    def row_coordinates(self) -> Any:
        pass

    @property
    def column_coordinates(self) -> Any:
        pass

    @property
    def row_contributions(self) -> Any:
        pass

    @property
    def column_contributions(self) -> Any:
        pass

    @property
    def row_cosine_similarities(self) -> Any:
        pass

    @property
    def column_cosine_similarities(self) -> Any:
        pass

    def plot_coordinates(self) -> Any:
        pass

    def scree_plot(self) -> Any:
        pass


class MCA_v2(MCA):
    """Run MCA analysis.

    V2 is based on `prince` package.
    GitHub Repo -> https://github.com/MaxHalford/prince
    """

    def __init__(self, data: pd.DataFrame, **kwargs) -> None:
        super().__init__(data, **kwargs)

    def _run_mca(self, **kwargs) -> Any:
        mca = prince.MCA(**kwargs)
        mca.fit(self.data)
        return mca

    @property
    def eigenvalues(self) -> Any:
        return self.obj.eigenvalues_

    @property
    def row_coordinates(self) -> Any:
        return self.obj.row_coordinates(self.data)

    @property
    def column_coordinates(self) -> Any:
        return self.obj.column_coordinates(self.data)
    
    @property
    def row_contributions(self) -> Any:
        return self.obj.row_contributions_.style.format('{:.0%}')
    
    @property
    def column_contributions(self) -> Any:
        return self.obj.column_contributions_.style.format('{:.0%}')
    
    @property
    def row_cosine_similarities(self) -> Any:
        return self.obj.row_cosine_similarities(self.data)
    
    @property
    def column_cosine_similarities(self) -> Any:
        return self.obj.column_cosine_similarities(self.data)

    def plot_coordinates(self) -> Any:
        return self.obj.plot(
            self.data,
            x_component=0,
            y_component=1,
            show_column_markers=True,
            show_row_markers=True,
            show_column_labels=False,
            show_row_labels=False,
        )
    
    def scree_plot(self, title: Optional[str] = "") -> Any:
        fig, ax = plt.subplots(figsize=(5, 4))
        # Set ticks to 0,1,..., until length of eigenvalues
        ax.bar(np.arange(1, len(self.eigenvalues)+1), self.eigenvalues, width=1, edgecolor="white", linewidth=0.7)
        ax.set_xlabel("Factor")
        ax.set_xticks(np.arange(1, len(self.eigenvalues)+1))
        ax.set_ylabel("Eigenvalue")
        title = f" of {title}" if title else ""
        ax.set_title(f"Scree plot {title}")
        plt.show()

    def scree_plot_fun(self, title: Optional[str] = "") -> Any:
        _title = f" of {title}" if title else ""
        def _scree_plot_fun():
            return self.scree_plot(title=_title)
        return _scree_plot_fun

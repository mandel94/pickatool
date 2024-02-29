from mca import MCA as _MCA
from typing import Union, TypeAlias, Any, Literal, NamedTuple, Optional, Iterable


class MCA(_MCA):
    """Run MCA analysis on a dataset.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.
    benzecri_correction : bool, default=True
        Whether to apply the Benzecri correction to the eigenvalues.
    TOL : float, default=1e-5
        Tolerance for the eigenvalues.

    Returns
    -------
    MCA
        An instance of the MCA class.
    """

    def __init__(self, data) -> None:
        super().__init__(self.data)
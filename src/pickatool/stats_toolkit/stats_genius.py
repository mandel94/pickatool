from abc import ABC, abstractmethod
from enum import Enum
from utils.types import DataFrame


class AvailableStatsEngines(Enum):
    V0 = "v0"


class MbaAlgorithms(Enum):
    APRIORI = "apriori"
    FP_GROWTH = "fp_growth"

class MbaParams:
    def __init__(self, data: DataFrame, min_support: float, min_confidence: float):
        self.data = data
        self.min_support = min_support
        self.min_confidence = min_confidence
    

def _apriori(params: MbaParams):
    # TODO Implement https://towardsdatascience.com/apriori-association-rule-mining-explanation-and-python-implementation-290b42afdfc6
    raise NotImplementedError("Apriori not implemented yet")


def _fp_growth(params: MbaParams):
    raise NotImplementedError("FP-Growth not implemented yet")


MbaAlgorithmsImpl: dict[MbaAlgorithms, callable] = {
    MbaAlgorithms.APRIORI: _apriori,
    MbaAlgorithms.FP_GROWTH: _fp_growth,
}


class _StatsGenius(ABC):

    @staticmethod
    def _get_mba_algorithm(algorithm: MbaAlgorithms):
        return MbaAlgorithmsImpl[algorithm]

    def __init__(self, data: DataFrame, engine: AvailableStatsEngines = "v0"):
        self.data = data
        self.engine = AvailableStatsEngines[engine.upper()]

    @abstractmethod
    def market_basket_analysis(self, algorithm: MbaAlgorithms = "apriori"):
        pass


def _init_stats_genius(data: DataFrame, engine: AvailableStatsEngines):
     return _StatsGenius(data, engine.upper())



class StatsGenius(_StatsGenius):

    def __init__(self, data: DataFrame, engine: AvailableStatsEngines = "v0"):
        super().__init__(data, engine)
        self._stats_genius = _init_stats_genius(data, engine) # The stats genius lying under the hood

    def market_basket_analysis(self, algorithm: MbaAlgorithms = "apriori"):
        return self._stats_genius._market_basket_analysis(algorithm)


class V0StatsGenius(_StatsGenius):

    def __init__(self, data: DataFrame):
        super().__init__(data, engine="v0")

    def _market_basket_analysis(self, algorithm: MbaAlgorithms = "apriori"):
        mba_algorithm = _StatsGenius._get_mba_algorithm(algorithm)
        return mba_algorithm(self.data, min_support=0.1, min_confidence=0.1)


    




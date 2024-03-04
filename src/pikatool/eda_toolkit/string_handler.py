from fuzzywuzzy import fuzz  # type: ignore
from fuzzywuzzy import process  # type: ignore
from typing import (
    Literal,
    TypeAlias,
    Optional,
    Iterable,
    Union,
    Callable,
    TypeVar,
    Self,
)
from queue import Queue
from pandas import DataFrame

FuzzyMatchMethod = Literal[
    "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"
]

FuzzyMatchFunction = {
    "ratio": fuzz.ratio,
    "partial_ratio": fuzz.partial_ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
}

T = TypeVar("T")


class Pipeline:
    def __init__(
        self, initial_collection: Iterable[T], steps: Iterable[Callable]
    ) -> Iterable[T]:
        self.steps = Queue()
        self.collection = initial_collection
        [self.steps.put(step) for step in steps]

    def get_next(self) -> Callable:
        return self.steps.get()

    def apply_next(self, collection: Iterable[T]) -> Iterable[T]:
        return [self.get_next()(element) for element in collection]

    def apply_all(self) -> Iterable:
        while not self.steps.empty():
            self.collection = self.apply_next(self.collection)
        return self.collection

    def add_step(self, step: Callable) -> None:
        self.steps.put(step)


class StringCollectionHandler:
    def __init__(self, collection: Iterable[str]) -> None:
        self.collection = collection
        pass

    def extract_most_similar(
        self, target: str, method: FuzzyMatchMethod = "token_sort_ratio"
    ) -> Union[str | None]:
        """Use fuzzy matching to extract the most similar string from the collection."""
        most_similar, score, _ = process.extractOne(
            target, self.collection, scorer=FuzzyMatchFunction[method]
        )
        return most_similar if score > 70 else None

    def map_to_collection(
        self,
        target_collection: Iterable[str],
        method: FuzzyMatchMethod = "token_sort_ratio",
    ):
        """Map each element of the target collection to the most similar element in the collection."""
        collection_set = set(self.collection)

    def load_pipeline(self, pipeline: Pipeline) -> Pipeline:
        self.pipeline = pipeline

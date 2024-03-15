from typing import Union, Iterable, Any, TypeAlias, Optional, Self, TypeVar, Callable
import pandas as pd
import sys


class PipelineError(Exception):

    def __init__(self, message, name, input, function) -> None:
       
        super().__init__(message)


class PipelineStep:

    def __init__(self, input: Any, function: Callable) -> None:
        self.input = input
        self.function = function

    def apply(self) -> Any:
        try:
            return self.function(self.input)
        except Exception as e:
            print(f"""Error in pipeline step: {self.name}. Input: {self.input} could not be applied to function: {self.function}""",
                  file=sys.stderr)
            raise  # Re-raise the exception U_U ---> https://subscription.packtpub.com/book/programming/9781788293181/6/06lvl1sec64/re-raising-exceptions
            
    

class Pipeline:
    """https://hazelcast.com/glossary/data-pipeline/"""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.steps = []

    def run(self) -> pd.DataFrame:
        tmp = self.data
        for step in self.steps:
            tmp = step.apply()
        return self.data


class DataProcessor():
    """Apply atomically a series of operations to a DataFrame."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initializes the class with the data.
        
        Args:
            data (pd.DataFrame): The data to be processed
        """
        self.data = data
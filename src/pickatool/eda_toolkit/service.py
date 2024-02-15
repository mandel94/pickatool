from abc import ABC, abstractmethod
from typing import List
import datetime


class Command(ABC):
    @abstractmethod
    def execute(self):
        pass


class Message():
    def __init__(self, id: int, timestamp: str, content, str):
        self.id = id
        self.timestamp = timestamp
        self.content = content

    

class Event(Message):
    def __init__(self, event_name: str, event_type: str, timestamp: str, content: str):
        super().__init__(event_name, event_type, timestamp, content)
        # Event specific logic here


class TaskEvent(Event):
    def __init__(self, task_id: int, task_name: str, event_name:str, event_type, timestamp: str, content: str):
        super().__init__(event_name, event_type, timestamp, content)
        self.task_id = task_id
        self.task_name = task_name


class Task():
    def __init__(self, name: str) -> None:
        self.name = name
        self.start_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.id = hash(name+self.start_time)
        self.task_events: List[TaskEvent] = [
            TaskEvent(task_id=self.id,
                      task_name=self.name,
                      event_time=self.start_time,
                      content="Task Started")
            ]


def EventChannel():
    def __init__(self):
        self.events = []

    def append_event(self, event: Event):
        self.events.append(event)

    def get_events(self):
        return self.events
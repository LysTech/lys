import time
from abc import ABC, abstractmethod

""" This is for recording task data during sessions: 
    - a pong task executor returns events like "user moved paddle left" 
    - a perceived speech executor returns events like "user heard the phrase 'blah'"
"""

class Event(ABC):
    """Base class for events to be logged. Always includes a timestamp in to_dict()."""
    def to_dict(self) -> dict:
        """Return a dict with timestamp and event-specific data."""
        return {
            "timestamp": time.time(),
            "event": self._to_dict()
        }

    @abstractmethod
    def _to_dict(self) -> dict:
        """Return event-specific data as a dict."""
        pass

class PauseEvent(Event):
    """Event representing a pause in the session."""
    def _to_dict(self) -> dict:
        return {"event_type": "pause"}

class ResumeEvent(Event):
    """Event representing a resume in the session."""
    def _to_dict(self) -> dict:
        return {"event_type": "resume"} 

class StartEvent(Event):
    """Event representing the start of a session or task."""
    def _to_dict(self) -> dict:
        return {"event_type": "start"} 
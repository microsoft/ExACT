import json
import hashlib
from typing import Any, Optional
from dataclasses import dataclass
from browser_env import Trajectory


@dataclass
class TaskRecord:
    task_info: dict
    trajectory: Trajectory
    Q: list[float]
    Nsa: list[int]
    P: list[float]
    V_next: list[float]
    final_score: float
    est_final_score: float

    _rubric: str = ""  # used by ReinforcedValueFunctions
    _debates: Optional[list[dict]] = None  # used by ReinforcedDBTValueFunctions
    _loaded_config: Optional[dict] = None

    def __hash__(self) -> int:
        # simply convert the task_info to a hash
        if self._loaded_config is None:
            with open(self.task_info['config_file'], "r") as fread:
                config = json.load(fread)
            self._loaded_config = config
        
        start_url = ""
        if "start_url" in self._loaded_config:
            start_url = self._loaded_config["start_url"]

        hashable_task_info = {
            "task_id": self.task_info['task_id'],
            "intent": self.task_info['intent'],
            # "rubric": self._rubric,
            "start_url": start_url,
        }

        unique_str = json.dumps(hashable_task_info, sort_keys=True)
        hash_object = hashlib.md5(unique_str.encode())
        hash_int = int(hash_object.hexdigest(), 16)
        return hash_int

    def __eq__(self, other) -> bool:
        if not isinstance(other, TaskRecord):
            return False
        return hash(self) == hash(other)
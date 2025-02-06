import json
import hashlib
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TaskRecord:
    task_info: dict
    trajectory: list[dict]
    Q: list[float]
    Nsa: list[int]
    P: list[float]
    V_next: list[float]
    success: float
    est_success: float

    _additional_info: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        domain = self.task_info['related_apps']
        task_id = self.task_info['id']
        instruction = self.task_info['instruction']
        hashable_task_info = {
            "domain": domain,
            "task_id": task_id,
            "instruction": instruction,
        }

        unique_str = json.dumps(hashable_task_info, sort_keys=True)
        hash_object = hashlib.md5(unique_str.encode())
        hash_int = int(hash_object.hexdigest(), 16)
        return hash_int

    def __eq__(self, other) -> bool:
        if not isinstance(other, TaskRecord):
            return False
        return hash(self) == hash(other)
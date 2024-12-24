from typing import Union, Any
from typing import TypedDict
import numpy.typing as npt
import numpy as np
from src.envs.actions import Action
# legacy code
from browser_env.utils import StateInfo as OldStateInfo
from browser_env import Action as OldAction


Observation = str | npt.NDArray[np.uint8]


class StateInfo(TypedDict):
    observation: dict[str, Observation]
    info: dict[str, Any]


Trajectory = list[Union[StateInfo, Action, OldStateInfo, OldAction]]
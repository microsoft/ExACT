from abc import ABC, abstractmethod
import os



class BaseAgent(ABC):
    name: str = "BaseAgent"

    @property
    def obs_processor(self):
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def predict(self, instruction: str, obs: dict, search_metadata) -> tuple[str, list]:
        """returns a generated response, and the list of actions parsed from the generated response

        Args:
            instruction (str): _description_
            obs (dict): _description_
            search_metadata (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            tuple[str, list]: response, list of actions
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("This method should be implemented by the subclass")


class RAgentMixin(ABC):
    """enables reflective agents.
    Before every task begins, the agent will call on_task_start, and after every task ends, the agent will call on_task_end
    """
    name: str = "RAgentMixin"

    def __init__(
        self,
        db_path: str,
        **kwargs
    ):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, exist_ok=True)
        return

    @abstractmethod
    def on_task_start(self, task_info: dict, **kwargs) -> None:
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def on_task_end(self, actual_trajectory: list, **kwargs) -> None:
        raise NotImplementedError("This method should be implemented by the subclass")


class ResumableAgentMixin(ABC):
    """enables resumable agents.
    """
    name: str = "ResumableAgentMixin"

    @abstractmethod
    def resume_predict(self, instruction: str, obs: dict, search_metadata) -> tuple[str, list]:
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def save_state(self, save_path: str) -> None:
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def load_state(self, save_path: str) -> None:
        raise NotImplementedError("This method should be implemented by the subclass")
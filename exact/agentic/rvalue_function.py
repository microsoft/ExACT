from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
from cachetools import Cache
from exact.agent.base import RAgentMixin
from exact.llms import lm_config
from exact.llms.tokenizer import Tokenizer
from exact.agentic.types import TaskRecord
from exact.agentic.value_function import CoTValueArgs, CoTValueFunction, SingleDebateValueFunction
from exact.logging import time_it
import hashlib
import os
import numpy as np
import logging


logger = logging.getLogger("src.agentic")


@dataclass
class ValueReflectionRecord:
    pass


class ReinforcedValueFunctionMixin(RAgentMixin):
    def __init__(
        self,
        db_path: str | Path,
        embedding_config: lm_config.LMConfig,
        embedding_tokenizer: Tokenizer,
        rlm_config: lm_config.LMConfig,
        rlm_tokenizer: Tokenizer,
    ):
        self.db_path = db_path
        self.rlm_config = rlm_config
        self.rlm_tokenizer = rlm_tokenizer
        self.embedding_config = embedding_config
        self.embedding_tokenizer = embedding_tokenizer

        self._retrieval_cache = Cache(maxsize=1000)  # MAY be used later

        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, exist_ok=True)
        return

    def on_task_start(self, task_info: dict, **kwargs) -> None:
        """Called when the task start. Used for reinforced MCTS"""
        return

    def on_task_end(self, actual_trajectory: list, **kwargs) -> None:
        """Called when the task ends. Used for reinforced MCTS"""
        return

    def retrieve_reflections(self, curr_task_intent: str, curr_obs: dict) -> list[ValueReflectionRecord]:
        raise NotImplementedError


@dataclass
class NoopReinforcedCoTValueArgs(CoTValueArgs):
    value_func: str = field(
        default="noop_r_cot_value_func",
        metadata={"help": "The name of the value function."}
    )
    vf_db_path: str | Path = field(
        default=None,
        metadata={"help": "If none, will default to the root dir running the experiment"}
    )


class NoopReinforcedCoTValueFunction(CoTValueFunction, ReinforcedValueFunctionMixin):
    name: str = "noop_r_cot_value_func"
    def __init__(
        self,
        args: NoopReinforcedCoTValueArgs,
        observation_type: str,
        action_space: str,
    ):
        CoTValueFunction.__init__(self, args, observation_type, action_space)
        ReinforcedValueFunctionMixin.__init__(
            self,
            db_path=args.vf_db_path,
            embedding_config=None,
            embedding_tokenizer=None,
            rlm_config=None,
            rlm_tokenizer=None,
        )
        return

    @time_it
    def on_task_start(self, task_info: dict, **kwargs) -> None:
        return

    @time_it
    def on_task_end(
        self,
        actual_trajectory: list[dict],
        task_info: dict,
        meta_data: Any,
        task_record: TaskRecord,
    ) -> None:
        return

    def retrieve_reflections(self, curr_task_intent: str, curr_obs: dict) -> list[ValueReflectionRecord]:
        return []


class NoopReinforcedSingleDebateValueFunction(SingleDebateValueFunction, ReinforcedValueFunctionMixin):
    name: str = "noop_r_sad_value_func"
    def __init__(
        self,
        args: NoopReinforcedCoTValueArgs,
        observation_type: str,
        action_space: str,
    ):
        SingleDebateValueFunction.__init__(self, args, observation_type, action_space)
        ReinforcedValueFunctionMixin.__init__(
            self,
            db_path=args.vf_db_path,
            embedding_config=None,
            embedding_tokenizer=None,
            rlm_config=None,
            rlm_tokenizer=None,
        )
        return

    @time_it
    def on_task_start(self, task_info: dict, **kwargs) -> None:
        return

    @time_it
    def on_task_end(
        self,
        actual_trajectory: list[dict],
        task_info: dict,
        meta_data: Any,
        task_record: TaskRecord,
    ) -> None:
        return

    def retrieve_reflections(self, curr_task_intent: str, curr_obs: dict) -> list[ValueReflectionRecord]:
        return []
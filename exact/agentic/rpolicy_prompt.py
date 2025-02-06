from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
from cachetools import Cache
from hashlib import sha256
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from exact.logging import time_it
from exact.agentic.types import TaskRecord
from exact.agent.base import RAgentMixin
from exact.prompts.rpolicy_prompt import (
    RPOLICY_COT_ADDITIONAL_INTRO, RPOLICY_COT_ADDITIONAL_INTRO_W_REFL,
    RPOLICY_COT_REFLECTION_PROMPT, RPOLICY_COT_EXPECTATION_PROMPT
)
from exact.llms.lm_config import LMConfig
from exact.llms.tokenizer import Tokenizer
from exact.llms.utils import (
    configure_llm_client, call_llm, 
    _truncate_prompt_to_max_tokens, _force_truncate_prompt_to_max_tokens,
    _flatten_chat_msg_turns
)
from exact.agentic.policy_prompt import ReACTPolicy, ReACTPolicyArgs
from exact.agentic.retriever import FaissRetriever
import pickle
import lzma
import hashlib
import os
import numpy as np
import logging
import copy


logger = logging.getLogger("src.agentic")


@dataclass
class ReinforcedReACTPolicyArgs(ReACTPolicyArgs):
    name: str = "rreact"

    ### embedding and reflection model
    embedding_model: str = "text-embedding-3-small"
    embedding_api_provider: str = "openai"
    rlm_model: str = "gpt-4o"
    rlm_api_provider: str = "openai"
    rlm_temperature: float = 0.7
    rlm_top_p: float = 0.9
    rlm_max_tokens: int = 400
    rlm_force_context_truncation: bool = False
    rlm_flatten_chat_msg: bool = False
    rlm_flatten_engine: str = "vllm"
    rlm_max_context_length: int = 0

    ### reflection config
    db_path: str | Path = field(
        default=None,
        metadata={"help": "If none, will default to the root dir running the experiment"}
    )
    selection_metric: str = field(
        default="unexpected_score",
        metadata={
            "choices": ["unexpected_score", "unexpected_and_absq"],
            "help": "How to select the action to reflect on"
        }
    )
    max_reflections_per_task: int = 2
    reflection_threshold: float = 0.5
    min_retrieval_score: float = 0.25
    max_to_retrieve: int = 2
    use_gt_success: bool = field(
        default=False,
        metadata={"help": "Whether to use ground truth success of estimated success for EOT reward"}
    )


@dataclass
class PolicyReflectionRecord:
    # important!
    instruction: str
    state_str: str
    state_img_arr: np.ndarray
    response_str: str
    next_state_str: str
    next_state_img_arr: Optional[np.ndarray]
    reflection: str
    _from_task_hash: str  # map back to the task record

    def __hash__(self):
        # !! python's built-in hash() is not deterministic
        unique_str = self.instruction + self.state_str + self.response_str + self.next_state_str + self.reflection
        hash_object = hashlib.md5(unique_str.encode())
        hash_int = int(hash_object.hexdigest(), 16)
        return hash_int

    def simplified_info(self) -> dict:
        return {
            "instruction": self.instruction,
            "reflection": self.reflection,
            "hash": hash(self),
        }

    def __eq__(self, other):
        if not isinstance(other, PolicyReflectionRecord):
            return False
        return hash(self) == hash(other)


class ReinforcedPolicyMixin(RAgentMixin):
    def __init__(
        self,
        db_path: str | Path,
        embedding_config: LMConfig,
        embedding_tokenizer: Tokenizer,
        rlm_config: LMConfig,
        rlm_tokenizer: Tokenizer,
    ):
        RAgentMixin.__init__(self, db_path)
        self.rlm_config = rlm_config
        self.rlm_tokenizer = rlm_tokenizer
        self.embedding_config = embedding_config
        self.embedding_tokenizer = embedding_tokenizer

        self._retrieval_cache = Cache(maxsize=1000)  # MAY be used later
        return

    def on_task_start(self, task_info: dict, **kwargs) -> None:
        """Called when the task start. Used for reinforced MCTS"""
        return

    def on_task_end(self, actual_trajectory: list, **kwargs) -> None:
        """Called when the task ends. Used for reinforced MCTS"""
        return

    def retrieve_reflections(self, curr_task_intent: str, curr_obs: dict) -> list[PolicyReflectionRecord]:
        raise NotImplementedError


def get_rpolicy_retrieval_key(instruction: str, obs: dict):
    obs_text = obs["accessibility_tree"]
    obs_image = obs["screenshot"]
    assert obs_text or obs_image, "at least one of obs_text or obs_image should be present"

    if obs_text is None:
        obs_text = ''

    if obs_image is None:
        obs_image_hash = ''
    elif isinstance(obs_image, str):
        obs_image_hash = obs_image
    else:
        obs_image_hash = sha256(obs_image).hexdigest()
    return f"{instruction}__{obs_text}__{obs_image_hash}"


class ReinforcedReACTPolicy(ReACTPolicy, ReinforcedPolicyMixin):
    """+ perform reflection lookup before every action, and does reflection at the end of every task"""

    def __init__(
        self,
        args: ReinforcedReACTPolicyArgs,
        system_message,
        observation_type: str,
        action_space: str,
        policy_lm_config: LMConfig,
    ):
        # lm is the agent, rlm is used to generate reflection, embedding is used for retrieval
        ReACTPolicy.__init__(self, args, system_message, observation_type, action_space)
        self.args = args
        self.action_space = action_space
        embedding_config, embedding_client = self._configure_single_client(
            model_name=args.embedding_model,
            model_api_provider=args.embedding_api_provider,
            llm_type="EMBEDDING",
            temperature=0.0,
            top_p=0.0,
            max_tokens=0,
        )
        rlm_config, rlm_client = self._configure_single_client(
            model_name=args.rlm_model,
            model_api_provider=args.rlm_api_provider,
            llm_type="REFLECTION_LLM",
            temperature=args.rlm_temperature,
            top_p=args.rlm_top_p,
            max_tokens=args.rlm_max_tokens,
            max_context_length=args.rlm_max_context_length,
        )

        ReinforcedPolicyMixin.__init__(
            self, args.db_path,
            embedding_config, embedding_config.tokenizer_cls,
            rlm_config, rlm_config.tokenizer_cls
        )
        
        self.policy_lm_tokenizer = policy_lm_config.tokenizer_cls

        self.embedding_client = embedding_client
        self.rlm_client = rlm_client

        self.reflection_folder_path = os.path.join(self.db_path, "policy_reflections")
        if not os.path.exists(self.reflection_folder_path):
            os.makedirs(self.reflection_folder_path, exist_ok=True)
        self.reflection_index_path = os.path.join(self.db_path, "policy_reflections_index")
        self.all_reflections = []
        self._reflection_prompt = RPOLICY_COT_REFLECTION_PROMPT

        self._task_record_folder_path = os.path.join(self.db_path, "task_records")
        if not os.path.exists(self._task_record_folder_path):
            os.makedirs(self._task_record_folder_path, exist_ok=True)
        self._all_task_records = []
        self._dochash_to_record = {}
        self.retriever = None
        return

    @staticmethod
    def _configure_single_client(
        model_name: str,
        model_api_provider: str,
        llm_type: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        max_context_length: int = 0,
    ):
        assert llm_type in ["REFLECTION_LLM", "EMBEDDING"]

        if "LLM" in llm_type:
            config = LMConfig(
                provider=model_api_provider,
                model=model_name,
                mode="chat",
                tokenizer_cls=Tokenizer(
                    provider=model_api_provider,
                    model_name=model_name,
                    max_context_length=max_context_length
                ),
                api_base=os.environ.get(f"{llm_type}_API_BASE", "http://127.0.0.1:30000/v1"),
                api_key=os.environ.get(f"{llm_type}_API_KEY", "empty"),
                api_version=os.environ.get(f"{llm_type}_API_VERSION", ""),
                api_token_provider_base=os.environ.get(f"{llm_type}_TOKEN_PROVIDER_BASE", ""),
                gen_config={
                    'temperature': temperature,
                    'top_p': top_p,
                    'max_tokens': max_tokens,
                }
            )
            client = configure_llm_client(config)
            return config, client
        elif "EMBEDDING" in llm_type:
            config = LMConfig(
                provider=model_api_provider,
                model=model_name,
                mode="chat",
                tokenizer_cls=Tokenizer(
                    provider=model_api_provider,
                    model_name=model_name
                )
            )
            if model_api_provider == "openai":
                client = OpenAIEmbeddings(
                    model=model_name,
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                    organization=os.environ.get("OPENAI_ORGANIZATION", ""),
                )
            elif model_api_provider == "sglang":
                client = OpenAIEmbeddings(
                    model=model_name,
                    base_url=os.environ.get(f"{llm_type}_API_BASE", "http://127.0.0.1:30100/v1"),
                    api_key=os.environ.get(f"{llm_type}_API_KEY", "empty"),
                )
            else:
                raise ValueError(f"Invalid model_api_provider {model_api_provider}")
            return config, client
        else:
            raise ValueError(f"Invalid llm_type {llm_type}")

    def _load_lzma_db_files(self, folder_path):
        loaded_data = []
        for file in os.listdir(folder_path):
            if file.endswith(".pkl.xz"):
                file_path = os.path.join(folder_path, file)
                try:
                    with lzma.open(file_path, "rb") as fread:
                        data = pickle.load(fread)
                    loaded_data.append(data)
                except Exception as e:
                    logger.error(e, exc_info=True)
                    logger.error(f"Error loading {file_path}. Maybe other thread is running write.")
        return loaded_data

    def _write_lzma_db_files(self, folder_path, hashable_data: list):
        for data in hashable_data:
            file_path = os.path.join(folder_path, f"{hash(data)}.pkl.xz")
            with lzma.open(file_path, "wb") as fwrite:
                pickle.dump(data, fwrite)
        return

    def _load_db(self):
        all_reflections: list[PolicyReflectionRecord] = self._load_lzma_db_files(self.reflection_folder_path)
        logger.info(f"Loaded {len(all_reflections)} reflection records from {self.reflection_folder_path}")
        deduped_reflections = list(set(all_reflections))
        if len(all_reflections) != len(deduped_reflections):
            logger.warning(f"Found {len(all_reflections) - len(deduped_reflections)} duplicates in reflection records.")
        self.all_reflections = deduped_reflections

        ### embedding
        embedding_docs = []
        for i, record in enumerate(deduped_reflections):
            task_intent = record.instruction
            curr_state = record.state_str

            doc_text = f"Task: {task_intent}\nObservation:\n{curr_state}"
            doc = Document(doc_text)
            embedding_docs.append(doc)

            doc_text_hashed = sha256(doc_text.encode()).hexdigest()
            self._dochash_to_record[doc_text_hashed] = record

        if len(embedding_docs) == 0:
            return None
        
        logger.debug("Initializing FaissRetriever")
        retriever = FaissRetriever(
            index_save_path=self.reflection_index_path,
            docs=embedding_docs,
            embeddings=self.embedding_client,
        )
        return retriever

    @time_it
    def call_llm(self, lm_config: LMConfig, client, messages, num_outputs=1):
        ## 1. truncate the prompt instead of simple left truncate if we let tokenizer do it
        if self.args.rlm_force_context_truncation:
            messages = _force_truncate_prompt_to_max_tokens(
                prompt=messages,
                tokenizer=lm_config.tokenizer_cls,
            )
        else:
            messages = _truncate_prompt_to_max_tokens(
                prompt=messages,
                tokenizer=lm_config.tokenizer_cls
            )
        ## 2. flatten the chat messages (useful if the models are also TRAINED this way)
        if self.args.rlm_flatten_chat_msg:
            engine = self.args.rlm_flatten_engine
            messages = _flatten_chat_msg_turns(messages, engine=engine)
        
        ## 3. call the model
        response = call_llm(
            client,
            lm_config,
            messages,
            num_outputs=num_outputs
        )
        return response

    @time_it
    def on_task_start(self, task_info: dict, **kwargs) -> None:
        self.retriever = self._load_db()
        return

    def _selection_metric(self, task_record: TaskRecord) -> float:
        expected_v = task_record.V_next  # VLM's guess of next state quality
        actual_Qsa = task_record.Q  # tree's guess of next state quality

        if self.args.selection_metric == "unexpected_score":
            unexpected_score = [abs(v - q) for v, q in zip(expected_v, actual_Qsa)]
        elif self.args.selection_metric == "unexpected_and_absq":
            # i.e., either action is super unspected, or expected but super good/bad
            unexpected_score = [abs(v - q) + abs(q) for v, q in zip(expected_v, actual_Qsa)]
        return unexpected_score

    def _get_action_str(self, action_dict):
        ### use parsed actions instead of raw_action to remove reasoning noises (e.g., 'based on reflection, etc')
        assert 'actions' in action_dict, f"action_dict should have 'actions' key, but has {action_dict.keys()}"
        if self.action_space == "pyautogui":
            actions = action_dict['actions']
            formatted_actions = [f"```python\n{action}\n```" for action in actions]
            action_str = "\nthen execute\n".join(formatted_actions)
        elif self.action_space == "computer_13":
            raise NotImplementedError("computer_13 action space not supported yet")
        else:
            raise ValueError(f"Invalid action_space: {self.action_space}")
        return action_str

    def _get_state_action_to_reflect(self, task_record: TaskRecord) -> tuple[dict, str, dict, dict]:
        action_scores = self._selection_metric(task_record)
        most_unexpected_idx = np.argmax(action_scores)

        logger.debug((
            f"most_unexpected_idx: {most_unexpected_idx}, score={action_scores[most_unexpected_idx]}"
        ))

        state = task_record.trajectory[most_unexpected_idx * 2]
        action = self._get_action_str(task_record.trajectory[most_unexpected_idx * 2 + 1])
        next_state = task_record.trajectory[most_unexpected_idx * 2 + 2]

        # other metadata that might be helpful
        all_actions_before_current = []
        picked_action_idx = most_unexpected_idx * 2 + 1
        if picked_action_idx > 1:
            for data in task_record.trajectory[:picked_action_idx]:
                if 'raw_action' in data:
                    action_str = self._get_action_str(data)
                    all_actions_before_current.append(action_str)
        else:
            all_actions_before_current = []

        if self.args.use_gt_success:
            success = task_record.success == 1.0 # RLHF typed
        else:
            success = task_record.est_success == 1.0  # self-supervised
        metadata = {
            "instruction": task_record.task_info["instruction"],
            "success": success,
            "all_actions_before_current": all_actions_before_current,
        }
        return state, action, next_state, metadata

    def _format_single_state(self, state: dict, prefix: str = "") -> dict:
        if prefix == "":
            prefix = " What's the next step that you will do to help with the task?"
        
        if self.observation_type == "screenshot":
            base64_image = state["screenshot"]

            formatted_content = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given the screenshot as below.{prefix}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        elif self.observation_type == "screenshot_a11y_tree":
            base64_image = state["screenshot"]
            linearized_accessibility_tree = state["accessibility_tree"]

            formatted_content = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given the screenshot and info from accessibility tree as below:\n{linearized_accessibility_tree}\n{prefix.strip()}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        elif self.observation_type == "a11y_tree":
            linearized_accessibility_tree = state["accessibility_tree"]

            formatted_content = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given the info from accessibility tree as below:\n{linearized_accessibility_tree}\n{prefix.strip()}"
                    }
                ]
            }
        elif self.observation_type == "som":
            base64_image = state["screenshot"]
            linearized_accessibility_tree = state["accessibility_tree"]

            formatted_content = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given the tagged screenshot and info from accessibility tree as below:\n{linearized_accessibility_tree}\n{prefix.strip()}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        else:
            raise ValueError("Invalid observation_type type: " + self.observation_type)
        return formatted_content

    def _get_expectation_prompt(
        self,
        intro: str,
        curr_state: dict,
        action_str: str,
    ):
        """Return the require format for an API"""
        messages = []
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": intro
                },
            ]
        })
        user_state_message = self._format_single_state(curr_state)
        messages.append(user_state_message)

        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": action_str.strip() if len(action_str) > 0 else "No valid action"
                },
            ]
        })
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": RPOLICY_COT_EXPECTATION_PROMPT
                },
            ]
        })
        return messages

    def _get_reflections_prompt(
        self,
        intro: str,
        curr_state: dict,
        action_str: str,
        expecatation_str: str,
        next_state: dict,
        metadata: dict
    ):
        messages = []
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": intro
                },
            ]
        })
        user_state_message = self._format_single_state(curr_state)
        
        #### add past agent action history, if any
        all_actions_before_current: list[str] = metadata["all_actions_before_current"]
        if len(all_actions_before_current) > 0:
            _str_list = ["PREVIOUS ACTIONS generated by the agent:"]
            for a_idx, a in enumerate(all_actions_before_current):
                _str_list.append(f"[Action {a_idx+1}]:\n{a}")
            action_hist_str = "\n".join(_str_list)
            user_state_message['content'].insert(
                0, 
                {
                    "type": "text",
                    "text": action_hist_str,
                }
            )
        messages.append(user_state_message)

        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"{action_str}\n\nExpectation for next observation: {expecatation_str}",
                },
            ]
        })
        
        ### next state
        next_user_state_message = self._format_single_state(next_state, prefix="")  # we need custom questions now

        ### final question
        task_success = metadata['success']
        task_status_str = "INCORRECTLY finished" if not task_success else "successfully completed"
        next_user_state_message['content'].append({
            "type": "text",
            "text": self._reflection_prompt.format(
                instruction=metadata['instruction'],
                task_status_str=task_status_str
            )
        })
        messages.append(next_user_state_message)
        return messages

    def _get_reflections(self, state: dict, action: str, next_state: dict, metadata: dict) -> str:
        # prompt llm
        lm_config = self.rlm_config
        client = self.rlm_client

        # always the normal intro
        instruction = metadata["instruction"]
        sys_msg = self.system_message + "\nYou are asked to complete the following task: {}".format(instruction)
        sys_msg += "\n" + RPOLICY_COT_ADDITIONAL_INTRO

        ### 1. get expectation (forward pass)
        expectation_prompt = self._get_expectation_prompt(
            intro=sys_msg,
            curr_state=state,
            action_str=action,
        )
        expecatation_ans = self.call_llm(
            lm_config,
            client,
            expectation_prompt,
            num_outputs=1
        )

        ### 2. get reflections (backward pass)
        reflections_prompt = self._get_reflections_prompt(
            intro=sys_msg,
            curr_state=state,
            action_str=action,
            expecatation_str=expecatation_ans,
            next_state=next_state,
            metadata=metadata
        )
        gen_reflections = self.call_llm(
            lm_config,
            client,
            reflections_prompt,
            num_outputs=1
        )
        return gen_reflections

    def reflect(self, new_task_record: TaskRecord) -> list[PolicyReflectionRecord]:
        """Reflect on the task and return lessons learned"""
        max_num_records = self.args.max_reflections_per_task
        tmp_task_record: TaskRecord = copy.deepcopy(new_task_record)
        # adjust Q, V from (q, v_next, q, v_next) to (q, v_next, q, final score) for _selection_metric
        if self.args.use_gt_success:
            tmp_task_record.V_next[-1] = 1.0 if new_task_record.success else -1.0 # RLHF typed
        else:
            tmp_task_record.V_next[-1] = 1.0 if new_task_record.est_success else -1.0 # self-supervised
        reflection_records = []
        _metadata_to_save = []
        for i in range(max_num_records):
            logger.info(f"Reflecting iteration {i}")
            action_scores_ = self._selection_metric(tmp_task_record)
            most_unexpected_idx = np.argmax(action_scores_)
            most_unexpected_score = action_scores_[most_unexpected_idx]
            if most_unexpected_score < self.args.reflection_threshold:
                logger.info(f"unexpected score {most_unexpected_score} is below {self.args.reflection_threshold=}. Stopping reflection.")
                break

            state, action, next_state, metadata = self._get_state_action_to_reflect(tmp_task_record)
            reflection = self._get_reflections(state, action, next_state, metadata)
            logger.info(f"reinforced policy model generated reflection: {reflection}")

            new_reflection_record = PolicyReflectionRecord(
                instruction=new_task_record.task_info["instruction"],
                state_str=state["accessibility_tree"],
                state_img_arr=state["screenshot"],
                response_str=action,
                next_state_str=next_state["accessibility_tree"],
                next_state_img_arr=next_state["screenshot"],
                reflection=reflection,
                _from_task_hash=hash(new_task_record)
            )
            reflection_records.append(new_reflection_record)
            _metadata_to_save.append({
                "state_str": state["accessibility_tree"],
                "response_str": action,
                "reflection": reflection,
            })

            # pop the most unexpected state and action
            tmp_task_record.trajectory.pop(most_unexpected_idx * 2)
            tmp_task_record.trajectory.pop(most_unexpected_idx * 2)  # yes, to pop state and next action
            tmp_task_record.Q.pop(most_unexpected_idx)
            tmp_task_record.Nsa.pop(most_unexpected_idx)
            tmp_task_record.P.pop(most_unexpected_idx)
            tmp_task_record.V_next.pop(most_unexpected_idx)

            if len(tmp_task_record.trajectory) < 2:
                break

        ## add reflection text into the task record
        new_task_record._additional_info['r-react-policy_reflections'] = _metadata_to_save
        return reflection_records
    
    @time_it
    def retrieve_reflections(self, curr_task: str, curr_obs: dict) -> list[PolicyReflectionRecord]:
        if self.retriever is None:
            logger.debug("Retriever not initialized. Skipping retrieve_reflections")
            return []
        
        # lets make it pure text first
        obs_text = curr_obs["accessibility_tree"]

        query_str = f"Task: {curr_task}\nObservation:\n{obs_text}"
        relevant_reflections = self.retriever.retrieve(
            query_str,
            min_score=self.args.min_retrieval_score,
            k=self.args.max_to_retrieve
        )
        reflection_records = []
        for doc in relevant_reflections:
            content_hashed = sha256(doc.page_content.encode()).hexdigest()
            record: PolicyReflectionRecord = self._dochash_to_record[content_hashed]
            reflection_records.append(record)
        return reflection_records

    def _retrieve_reflections_to_prompt(self, curr_task: str, curr_obs: dict) -> str:
        reflection_records = self.retrieve_reflections(curr_task, curr_obs)
        retrieval_key = get_rpolicy_retrieval_key(curr_task, curr_obs)
        self._retrieval_cache[retrieval_key] = reflection_records  # internally used by RMCTS agent

        reflection_texts = []
        for r_i, record in enumerate(reflection_records):
            record: PolicyReflectionRecord

            instruction = record.instruction
            state_text = self.policy_lm_tokenizer.decode(self.policy_lm_tokenizer.encode(record.state_str)[:512])
            action_str = record.response_str
            next_state_text = self.policy_lm_tokenizer.decode(self.policy_lm_tokenizer.encode(record.next_state_str)[:200])
            reflection_text = self.policy_lm_tokenizer.decode(self.policy_lm_tokenizer.encode(record.reflection)[:128])
            
            reflection_texts.append((
                f"OBJECTIVE ({r_i+1}): {instruction}\n"
                # f"TRUNCATED OBSERVATION ({r_i+1}):\n{state_text}\n"
                f"ATTEMPTED ACTION ({r_i+1}): {action_str}\n"
                f"REFLECTION ({r_i+1}): {reflection_text}"
            ))

        instruction = '\n#####\n'.join(reflection_texts).strip()
        logger.info(f"constructed additional instruction from retrieval=\n{instruction}")
        return instruction

    def _insert_reflection_str_to_prompt(self, prompt: list[dict], reflections: str):
        ### 2.3 inject reflection into current state
        additional_instructions = None
        reflection_injects = None

        if reflections != '':
            reflection_injects = {
                "type": "text",
                "text": (
                    "REFLECTIONS: here are some relevant reflections from other tasks. "
                    "Note that while these reflections may NOT directly relates to the current task, they may provide some useful insights.\n"
                    f"[START OF REFLECTIONS]\n{reflections}\n[END OF REFLECTIONS]"
                )
            }
        # helper to ask them follow all the relevant materials
        if len(prompt) > 1:
            if reflections != '':
                additional_instructions = {
                    "type": "text",
                    "text": RPOLICY_COT_ADDITIONAL_INTRO_W_REFL + "\n" + "Remember to consider user's task, previous histories, "
                            "reflections (if applicable), and the above guidelines to better plan the next action."
                }
            else:
                additional_instructions = {
                    "type": "text",
                    "text": RPOLICY_COT_ADDITIONAL_INTRO + "\n" + "Remember to consider user's task, previous histories, "
                            "and the above guidelines to better plan the next action."
                }
        else:
            if reflections != '':
                additional_instructions = {
                    "type": "text",
                    "text": RPOLICY_COT_ADDITIONAL_INTRO_W_REFL + "\n" + "Remember to consider user's task, "
                            "reflections (if applicable), and the above guidelines to better plan the next action."
                }
            else:
                additional_instructions = {
                    "type": "text",
                    "text": RPOLICY_COT_ADDITIONAL_INTRO + "\n" + "Remember to consider user's task, "
                            "and the above guidelines to better plan the next action."
                }
        if additional_instructions is not None:
            ## add to last position
            assert prompt[-1]["role"] == "user", "last message should be user"
            prompt[-1]["content"].append(additional_instructions)
        if reflection_injects is not None:
            ## add it to the first position
            assert prompt[-1]["role"] == "user", "last message should be user"
            prompt[-1]["content"].insert(0, reflection_injects)
        return prompt

    @time_it
    def get_messages(
        self,
        instruction: str,
        past_obs: list[dict],
        past_actions: list,
        past_thoughts: list
    ):
        normal_policy_msg = super().get_messages(instruction, past_obs, past_actions, past_thoughts)

        # retrieve reflections
        relevant_reflections_str = self._retrieve_reflections_to_prompt(
            curr_task=instruction, curr_obs=past_obs[-1],
        )
        intro = self.system_message + "\nYou are asked to complete the following task: {}".format(instruction)
        
        ### 1. update system message to instruct it to use the reflection
        assert normal_policy_msg[0]["role"] == "system", "first message should be system"
        normal_policy_msg[0] = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": intro
                },
            ]
        }

        ### 2. add the reflection back to the prompt
        policy_msg_w_refl = self._insert_reflection_str_to_prompt(
            normal_policy_msg,
            reflections=relevant_reflections_str
        )

        logger.info(f"constructed prompt with len={len(policy_msg_w_refl)}")
        return policy_msg_w_refl

    @time_it
    def on_task_end(
        self,
        actual_trajectory: list[dict],
        task_info: dict,
        meta_data: Any,
        task_record: TaskRecord,
    ) -> None:
        reflection_records = self.reflect(task_record)

        # save reflection record
        all_reflections: list[PolicyReflectionRecord] = self._load_lzma_db_files(self.reflection_folder_path)
        logger.info(f"Found {len(all_reflections)} reflections from {self.reflection_folder_path}")
        existing_refl_hashes = set([hash(r) for r in all_reflections])
        new_refl_to_write = [r for r in reflection_records if hash(r) not in existing_refl_hashes]
        logger.info(f"Deduped and writing {len(new_refl_to_write)} new reflections to {self.reflection_folder_path}")
        self._write_lzma_db_files(self.reflection_folder_path, new_refl_to_write)
        return
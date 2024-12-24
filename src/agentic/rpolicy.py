import logging
import pickle
import lzma
import os
import numpy as np
import json
import copy
import time
import hashlib
from pathlib import Path
from typing import Any, Optional
from PIL import Image
from dataclasses import dataclass
from cachetools import Cache

from browser_env import Trajectory, ActionTypes
from browser_env.utils import StateInfo, pil_to_b64, pil_to_vertex, Observation
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.document import Document
from hashlib import sha256
from src.llms import lm_config
from src.llms.tokenizer import Tokenizer
from src.llms.utils import _add_modality_key_for_sglang_messages
from src.agentic.policy import MCoTPolicyPConstructor_OLD
from src.prompts.utils import FaissRetriever, display_multimodal_openai_messages
from src.prompts.types import TaskRecord
from src.llms import call_llm
from src.logging import time_it
from src.envs.actions import Action


logger = logging.getLogger("logger")


@dataclass
class ReflectionRecord:
    # important!
    intent: str
    state_str: str
    state_img_arr: np.ndarray
    action_str: str
    next_state_str: str
    next_state_img_arr: Optional[np.ndarray]
    reflection: str
    _from_task_hash: int  # map back to the task record

    def __hash__(self):
        # !! python's built-in hash() is not deterministic
        unique_str = self.intent + self.state_str + self.action_str + self.next_state_str
        hash_object = hashlib.md5(unique_str.encode())
        hash_int = int(hash_object.hexdigest(), 16)
        return hash_int

    def simplified_info(self) -> dict:
        return {
            "intent": self.intent,
            "reflection": self.reflection,
            "hash": hash(self),
        }

    def __eq__(self, other):
        if not isinstance(other, ReflectionRecord):
            return False
        return hash(self) == hash(other)


class ReinforcedPromptMixin:
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

        self._retrieval_cache = Cache(maxsize=100)  # MAY be used later

        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, exist_ok=True)
        return

    def on_task_start(self, task_info: dict, **kwargs) -> None:
        """Called when the task start. Used for reinforced MCTS"""
        return

    def on_task_end(self, trajectory: Trajectory, score: float, task_info: dict, meta_data: Any, **kwargs) -> None:
        """Called when the task ends. Used for reinforced MCTS"""
        return

    def retrieve_reflections(self, curr_task_intent: str, curr_obs: Observation) -> list[ReflectionRecord]:
        raise NotImplementedError



class ReinforcedPolicyPConstructor(MCoTPolicyPConstructor_OLD, ReinforcedPromptMixin):
    """+ perform reflection lookup before every action, and does reflection at the end of every task"""
    is_multimodal = True

    def __init__(
        self,
        instruction_path: str | Path,
        db_path : str | Path,
        lm_config: lm_config.LMConfig,
        rlm_config: lm_config.LMConfig,
        embedding_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
        rlm_tokenizer: Tokenizer,
        embedding_tokenizer: Tokenizer,
        ## behavioral args
        max_reflections_per_task: int = 1,
        reflection_threshold: float = 0.1,
        min_retrieval_score: float = 0.25,
        max_to_retrieve: int = 2,
        ## debugging
        use_gt_success: bool = False,
    ):
        # lm is the agent, rlm is used to generate reflection, embedding is used for retrieval
        MCoTPolicyPConstructor_OLD.__init__(self, instruction_path, lm_config, tokenizer)
        ReinforcedPromptMixin.__init__(
            self, db_path,
            embedding_config, embedding_tokenizer,
            rlm_config, rlm_tokenizer
        )
        assert 'agent_intro' in self.instruction, "agent_intro is required in the instruction for ReinforcedPrompts"
        assert 'intro_w_reflections' in self.instruction, "intro_w_reflections is required in the instruction for ReinforcedPrompts"

        self.embedding_model_name = self.embedding_config.model
        self.max_reflections_per_task = max_reflections_per_task
        self.reflection_threshold = reflection_threshold
        self.min_retrieval_score = min_retrieval_score
        self.max_to_retrieve = max_to_retrieve
        self.use_gt_success = use_gt_success
        self.reflection_folder_path = os.path.join(self.db_path, "policy_reflections")
        if not os.path.exists(self.reflection_folder_path):
            os.makedirs(self.reflection_folder_path, exist_ok=True)
        self.reflection_index_path = os.path.join(self.db_path, "policy_reflections_index")
        self.all_reflections = []

        self._task_record_folder_path = os.path.join(self.db_path, "task_records")
        if not os.path.exists(self._task_record_folder_path):
            os.makedirs(self._task_record_folder_path, exist_ok=True)
        self._all_task_records = []
        self._dochash_to_record = {}
        self.retriever = None
        return

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
        all_reflections: list[ReflectionRecord] = self._load_lzma_db_files(self.reflection_folder_path)
        logger.info(f"Loaded {len(all_reflections)} reflection records from {self.reflection_folder_path}")
        deduped_reflections = list(set(all_reflections))
        if len(all_reflections) != len(deduped_reflections):
            logger.warning(f"Found {len(all_reflections) - len(deduped_reflections)} duplicates in reflection records.")
        self.all_reflections = deduped_reflections

        ### embedding
        embedding_docs = []
        for i, record in enumerate(deduped_reflections):
            task_intent = record.intent
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
            embeddings=OpenAIEmbeddings(
                model=self.embedding_model_name,
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                organization=os.environ.get("OPENAI_ORGANIZATION", ""),
            )
        )
        return retriever

    def on_task_start(self, task_info: dict, **kwargs) -> None:
        self.retriever = self._load_db()
        return

    def _selection_metric(self, task_record: TaskRecord) -> float:
        expected_v = task_record.V_next
        actual_Qsa = task_record.Q
        unexpected_score = [abs(v - q) for v, q in zip(expected_v, actual_Qsa)]

        # skip none actions
        action_idx = 0
        for data in task_record.trajectory:
            if isinstance(data, Action):
                if action_idx >= len(unexpected_score):
                    break
                if data["action_type"] == ActionTypes.NONE:
                    # make that score negative infinity
                    unexpected_score[action_idx] = -float("inf")
                if ("failed to parse actions" in data.answer.lower()
                        or "error:" in data.answer.lower()
                        or "task terminated" in data.answer.lower()):
                    unexpected_score[action_idx] = -float("inf")  # skip these as well
                action_idx += 1
        return unexpected_score

    def _get_state_action_to_reflect(self, new_task_record: TaskRecord) -> tuple[StateInfo, Action, StateInfo, dict]:
        if self.use_gt_success:
            success = new_task_record.final_score > 0.0 # RLHF typed
        else:
            success = new_task_record.est_final_score > 0.0  # self-supervised
        action_scores = self._selection_metric(new_task_record)
        most_unexpected_idx = np.argmax(action_scores)

        logger.debug((
            f"most_unexpected_idx: {most_unexpected_idx}, score={action_scores[most_unexpected_idx]}"
        ))
        # if success:
        # else:

        state = new_task_record.trajectory[most_unexpected_idx * 2]
        action = new_task_record.trajectory[most_unexpected_idx * 2 + 1]
        if len(new_task_record.trajectory) > most_unexpected_idx * 2 + 2:
            next_state = new_task_record.trajectory[most_unexpected_idx * 2 + 2]
        else:
            if success:
                next_state = {
                    'is_terminated': True,
                    'observation': {
                        'text': "Evaluation: task completed successfully.",
                        'image': None
                    }
                }
            else:
                next_state = {
                    'is_terminated': True,
                    'observation': {
                        'text': "Evaluation: task failed.",
                        'image': None
                    }
                }

        # other metadata that might be helpful
        all_actions_before_current = []
        picked_action_idx = most_unexpected_idx * 2 + 1
        if picked_action_idx > 1:
            for data in new_task_record.trajectory[:picked_action_idx]:
                if isinstance(data, Action):
                    all_actions_before_current.append(data)
        else:
            all_actions_before_current = []
        metadata = {
            "intent": new_task_record.task_info["intent"],
            "intent_images": new_task_record.task_info["images"],
            "success": success,
            "final_score": new_task_record.final_score if self.use_gt_success else new_task_record.est_final_score,
            "all_actions_before_current": all_actions_before_current,
        }
        return state, action, next_state, metadata

    def _get_expectation_prompt(
        self,
        intro: str,
        curr_state_img: Image.Image,
        curr_state_text: str,
        action_str: str,
    ):
        """Return the require format for an API"""
        message: list[dict[str, str]] | str | list[str | Image.Image]
        if self.rlm_config.provider in ["openai", "sglang", "azure"]:
            if self.rlm_config.mode == "chat":
                message = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": intro}],
                    }
                ]
                expectation_prompt = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": f"OBSERVATION:\n{curr_state_text}"
                            },
                            {
                                "type": "text",
                                "text": "IMAGES: (1) current page screenshot",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": pil_to_b64(curr_state_img)},
                            },
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Action: {action_str}",
                            },
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "What do you expect to happen after taking this action? "
                                    "Briefly describe what you think will appear on the webpage after performing the action."
                                )
                            },
                        ]
                    }
                ]
                message.extend(expectation_prompt)

                if self.rlm_config.provider == "sglang":
                    message = _add_modality_key_for_sglang_messages(message)
                return message
        elif "google" in self.lm_config.provider:
            raise ValueError(
                f"Gemini models do not support yet"
            )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

    def _get_reflections_prompt(
        self,
        intro: str,
        curr_state_img: Image.Image,
        curr_state_text: str,
        action_str: str,
        expecatation_str: str,
        next_state_text: str,
        next_state_img: Optional[Image.Image],
        metadata: dict
    ):
        task_success = metadata['success']
        task_status_str = "failed" if not task_success else "successfully completed"

        message: list[dict[str, str]] | list[str | Image.Image] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": intro}],
            }
        ]
        all_actions_before_current: list[Action] = metadata["all_actions_before_current"]
        if self.rlm_config.provider in ["openai", "sglang", "azure"]:
            if self.rlm_config.mode == "chat":
                user_start_content = [
                    {
                        "type": "text", 
                        "text": f"OBSERVATION:\n{curr_state_text}"
                    },
                    {
                        "type": "text",
                        "text": "IMAGES: (1) current page screenshot",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(curr_state_img)},
                    },
                    {
                        "type": "text", 
                        "text": f"OBJECTIVE: {metadata['intent']}"
                    },
                ]
                if len(all_actions_before_current) > 0:
                    formatted_action_history = "PREVIOUS ACTIONS taken by the agent:\n"
                    for a_idx, a in enumerate(all_actions_before_current):
                        formatted_action_history += f"({a_idx+1}) {a.raw_prediction}\n"
                    formatted_action_history = formatted_action_history.strip()
                    user_start_content.insert(
                        0, 
                        {
                            "type": "text",
                            "text": formatted_action_history,
                        }
                    )
                if metadata["intent_images"] is not None:
                    for image_i, image in enumerate(metadata["intent_images"]):
                        user_start_content.extend(
                            [
                                {
                                    "type": "text",
                                    "text": f"({image_i+2}) objective input image {image_i+1}",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": pil_to_b64(image)},
                                },
                            ]
                        )

                reflection_prompt = [
                    {
                        "role": "user",
                        "content": user_start_content
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Action: {action_str}\n\nExpectation for next observation: {expecatation_str}",
                            },
                        ]
                    }
                ]
                #### now we ask for reflection
                # next_state_img is None when we have a termination
                if next_state_img is not None:
                    reflection_prompt.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": f"NEXT OBSERVATION:\n{next_state_text}"
                            },
                            {
                                "type": "text",
                                "text": "IMAGES: (1) next page screenshot",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": pil_to_b64(next_state_img)},
                            },
                            {
                                "type": "text", 
                                "text": (
                                    "Is this webpage what you expected? If not, can you conclude anything special about navigating on this website? "
                                    "If you faced the same situation again, what would you do differently at a high level? Do NOT propose any specific actions/answers.\n"
                                    "Keep your response within 100 words. "
                                    f"Note that according to our evaluation, you have {task_status_str} the OBJECTIVE at the very end."
                                )
                            },
                        ]
                    })
                else:
                    reflection_prompt.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": f"NEXT OBSERVATION:\n{next_state_text}"
                            },
                            {
                                "type": "text", 
                                "text": (
                                    "Is this result what you expected? If not, can you conclude anything special about navigating on this website? "
                                    "If you faced the same situation again, what would you do differently at a high level? Do NOT propose any specific actions/answers.\n"
                                    "Keep your response within 100 words. "
                                    f"Note that according to our evaluation, you have {task_status_str} the OBJECTIVE at the very end."
                                )
                            },
                        ]
                    })
                message.extend(reflection_prompt)

                if self.rlm_config.provider == "sglang":
                    message = _add_modality_key_for_sglang_messages(message)
                return message
        elif "google" in self.lm_config.provider:
            raise ValueError(
                f"Gemini models do not support yet"
            )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

    def _get_reflections(self, state: StateInfo, action: Action, next_state: StateInfo, metadata: dict) -> str:
        # prompt llm
        lm_config = self.rlm_config
        
        state_screenshot_arr = state["observation"]["image"]
        state_screenshot_img = Image.fromarray(state_screenshot_arr)
        state_text = state["observation"]["text"]
        if "is_terminated" in next_state:
            # terminated
            next_state_screenshot_img = None
        else:
            next_state_screenshot_arr = next_state["observation"]["image"]
            next_state_screenshot_img = Image.fromarray(next_state_screenshot_arr)
        next_state_text = next_state["observation"]["text"]

        # format prompt
        expectation_prompt = self._get_expectation_prompt(
            intro=self.instruction["agent_intro"],
            curr_state_img=state_screenshot_img,
            curr_state_text=state_text,
            action_str=action.raw_prediction,
        )
        # this will use agent LLM API
        expecatation_ans = call_llm(
            lm_config,
            expectation_prompt,
            num_outputs=1
        )

        reflections_prompt = self._get_reflections_prompt(
            intro=self.instruction["agent_intro"],
            curr_state_img=state_screenshot_img,
            curr_state_text=state_text,
            action_str=action.raw_prediction,
            expecatation_str=expecatation_ans,
            next_state_text=next_state_text,
            next_state_img=next_state_screenshot_img,
            metadata=metadata
        )
        gen_reflections = call_llm(
            lm_config,
            reflections_prompt,
            num_outputs=1
        )
        return gen_reflections

    def reflect(self, new_task_record: TaskRecord) -> list[ReflectionRecord]:
        """Reflect on the task and return lessons learned"""
        max_num_records = self.max_reflections_per_task
        tmp_task_record = copy.deepcopy(new_task_record)
        # adjust Q, V from (q, v_next, q, v_next) to (q, v_next, q, final score) for _selection_metric
        if self.use_gt_success:
            tmp_task_record.V_next[-1] = new_task_record.final_score # RLHF typed
        else:
            tmp_task_record.V_next[-1] = new_task_record.est_final_score
        reflection_records = []
        for i in range(max_num_records):
            logger.info(f"Reflecting iteration {i}")
            action_scores_ = self._selection_metric(tmp_task_record)
            most_unexpected_idx = np.argmax(action_scores_)
            most_unexpected_score = action_scores_[most_unexpected_idx]
            if most_unexpected_score < self.reflection_threshold:
                logger.info(f"unexpected score {most_unexpected_score} is below {self.reflection_threshold=}. Stopping reflection.")
                break

            state, action, next_state, metadata = self._get_state_action_to_reflect(tmp_task_record)
            reflection = self._get_reflections(state, action, next_state, metadata)
            logger.info(f"reinforced policy model generated reflection: {reflection}")

            new_reflection_record = ReflectionRecord(
                intent=new_task_record.task_info["intent"],
                state_str=state["observation"]["text"],
                state_img_arr=state["observation"]["image"],
                action_str=action.raw_prediction,
                next_state_str=next_state["observation"]["text"],
                next_state_img_arr=next_state["observation"]["image"],
                reflection=reflection,
                _from_task_hash=hash(new_task_record)
            )
            reflection_records.append(new_reflection_record)

            # pop the most unexpected state and action
            tmp_task_record.trajectory.pop(most_unexpected_idx * 2)
            tmp_task_record.trajectory.pop(most_unexpected_idx * 2)  # yes, to pop state and next action
            tmp_task_record.Q.pop(most_unexpected_idx)
            tmp_task_record.Nsa.pop(most_unexpected_idx)
            tmp_task_record.P.pop(most_unexpected_idx)
            tmp_task_record.V_next.pop(most_unexpected_idx)

            if len(tmp_task_record.trajectory) < 2:
                break
        return reflection_records
    
    def retrieve_reflections(self, curr_task_intent: str, curr_obs: Observation) -> list[ReflectionRecord]:
        if self.retriever is None:
            logger.debug("Retriever not initialized. Skipping retrieve_reflections")
            return []
        
        # lets make it pure text first
        obs_text = curr_obs["text"]

        query_str = f"Task: {curr_task_intent}\nObservation:\n{obs_text}"
        relevant_reflections = self.retriever.retrieve(
            query_str,
            min_score=self.min_retrieval_score,
            k=self.max_to_retrieve
        )
        reflection_records = []
        for doc in relevant_reflections:
            content_hashed = sha256(doc.page_content.encode()).hexdigest()
            record: ReflectionRecord = self._dochash_to_record[content_hashed]
            reflection_records.append(record)
        return reflection_records

    def _construct_context_specific_instruction(self, curr_task_intent: str, curr_obs: Observation) -> str:
        reflection_records = self.retrieve_reflections(curr_task_intent, curr_obs)
        self._retrieval_cache[(curr_task_intent, curr_obs["text"])] = reflection_records  # internally used by RMCTS agent

        reflection_texts = []
        for r_i, record in enumerate(reflection_records):
            record: ReflectionRecord

            intent = record.intent
            state_text = self.tokenizer.decode(self.tokenizer.encode(record.state_str)[:400])
            action_str = record.action_str
            next_state_text = self.tokenizer.decode(self.tokenizer.encode(record.next_state_str)[:200])
            reflection_text = self.tokenizer.decode(self.tokenizer.encode(record.reflection)[:128])
            
            reflection_texts.append((
                f"OBJECTIVE ({r_i+1}): {intent}\n"
                f"ATTEMPTED ACTION ({r_i+1}): {action_str}\n"
                f"REFLECTION ({r_i+1}): {reflection_text}"
            ))

        instruction = '\n#####\n'.join(reflection_texts).strip()
        logger.info(f"constructed context-specific instruction=\n{instruction}")
        return instruction
    
    def _is_long_history(self, all_prev_state_actions: Trajectory) -> bool:
        return len(all_prev_state_actions) >= 5 # (s,a,s,a,s)

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
        meta_data: dict[str, Any] = {},
    ):
        # simply format prompt using the FULL trajectory
        intro = self.instruction["intro"]
        intro_w_reflections = self.instruction["intro_w_reflections"]
        intro_wo_icl = self.instruction["intro_wo_icl"]
        examples = self.instruction["examples"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        images = images or []

        context_specific_instruction = self._construct_context_specific_instruction(
            curr_task_intent=intent, curr_obs=state_info["observation"],
        )
        if context_specific_instruction != "":
            intro = intro_w_reflections
        if self._is_long_history(trajectory):
            intro = intro_wo_icl
        
        ### format all past actions
        none_padded_action_history_str = copy.deepcopy(meta_data["action_history"])
        if none_padded_action_history_str[0].lower() != "none":
            none_padded_action_history_str.insert(0, "None")

        prompt = self.get_lm_api_input(
            intro, examples,
            intent=intent,
            intent_image=images,
            all_prev_state_actions=trajectory,
            all_prev_action_strs=none_padded_action_history_str,
            context_specific_instruction=context_specific_instruction,
        )
        logger.info(f"constructed prompt with len={len(prompt)}")
        logger.debug(f"constructed full prompt:\n{display_multimodal_openai_messages(prompt)}")
        return prompt

    def get_lm_api_input(
        self,
        intro: str,
        examples: list[tuple[str, str, str]],
        intent: str,
        intent_image: list[Image.Image],
        all_prev_state_actions: Trajectory,
        all_prev_action_strs: list[str],
        context_specific_instruction: str,
    ):
        """Return the require format for an API"""
        assert self.lm_config.provider in ["openai", "sglang", "azure"], f"Provider {self.lm_config.provider} not implemented"
        assert self.lm_config.mode == "chat", f"Mode {self.lm_config.mode} not implemented"
        assert len(all_prev_state_actions) % 2 == 1, f"all_prev_state_actions should be in (s,a, ...,s,a,s), but got {len(all_prev_state_actions)=}"
        message: list[dict[str, str]] | str | list[str | Image.Image]

        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": intro}],
            }
        ]

        ### 1. add few shot examples
        if self._is_long_history(all_prev_state_actions):
            examples = []  # no need
        for (x, y, z) in examples:
            example_img = Image.open(z)
            message.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": x},
                        {
                            "type": "text",
                            "text": "IMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": pil_to_b64(example_img)
                            },
                        },
                    ],
                }
            )
            message.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": y}],
                }
            )
        ### 2. add trajectory history
        ### 2.1 format init state
        history_prompt = []
        template = self.instruction["template"]
        init_template = self.instruction["init_template"]

        init_s = all_prev_state_actions[0]
        init_state_img = Image.fromarray(init_s["observation"]["image"])
        init_state_text = init_s["observation"]["text"]
        init_url = init_s["info"]["page"].url
        init_prev_action_str = all_prev_action_strs[0]
        hist_init = init_template.format(
            objective=intent,
            url=self.map_url_to_real(init_url),
            observation=init_state_text,
            previous_action=init_prev_action_str,
        )
        hist_init = (
            f"!IMPORTANT! Below is the task you need to solve. Please read the user's intent and input images (if any) carefully.\n"
            f"{hist_init}"
        )
        history_prompt.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": hist_init
                },
            ]
        })
        for image_i, image in enumerate(intent_image):
            history_prompt[0]["content"].extend(
                [
                    {
                        "type": "text",
                        "text": f"({image_i+1}) input image {image_i+1}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(image)},
                    },
                ]
            )
        history_prompt[0]["content"].extend([
            {
                "type": "text",
                "text": "IMAGES: (1) current page screenshot",
            },
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(init_state_img)},
            }
        ])
        ### 2.2 format history actions
        past_action_idx = 0
        # skip first state. This should end with the last/current state
        for s_or_a in all_prev_state_actions[1:]:
            if isinstance(s_or_a, Action):
                history_prompt.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{s_or_a.raw_prediction}"
                            },
                        ]
                    }
                )
                past_action_idx += 1
            else:
                state_img = Image.fromarray(s_or_a["observation"]["image"])
                state_text = s_or_a["observation"]["text"]
                page = s_or_a["info"]["page"]
                url = page.url
                previous_action_str = all_prev_action_strs[past_action_idx]
                hist_current = template.format(
                    url=self.map_url_to_real(url),
                    observation=state_text,
                    previous_action=previous_action_str,
                )
                history_prompt.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": hist_current
                            },
                            {
                                "type": "text",
                                "text": "IMAGES: (1) current page screenshot",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": pil_to_b64(state_img)},
                            },
                        ]
                    }
                )
        ### 2.3 inject reflection into current state
        additional_instructions = None
        reflection_injects = None
        if len(history_prompt) > 1:
            if context_specific_instruction != '':
                additional_instructions = (
                    "\nNOTE: Remember that you should consider user's intent, previous histories, "
                    "and reflections (if applicable) to better plan the next action."
                )
                reflection_injects = {
                    "type": "text",
                    "text": (
                        "REFLECTIONS: here are some relevant reflections from other tasks. "
                        "Note that these reflections may not be directly related to the new task below, but they may provide some useful insights.\n"
                        f"{context_specific_instruction}"
                    )
                }
            else:
                additional_instructions = (
                    "\nNOTE: Remember that you should consider user's intent and previous histories "
                    "to better plan the next action."
                )
        else:
            if context_specific_instruction != '':
                additional_instructions = (
                    "\nNOTE: Remember that you should consider user's intent and reflections (if applicable) "
                    "to better plan the next action."
                )
                reflection_injects = {
                    "type": "text",
                    "text": (
                        "REFLECTIONS: here are some relevant reflections from other tasks. "
                        "Note that these reflections may not be directly related to the new task below, but they may provide some useful insights.\n"
                        f"{context_specific_instruction}"
                    )
                }
        if additional_instructions is not None:
            history_prompt[-1]["content"][0]["text"] += additional_instructions
        if reflection_injects is not None:
            history_prompt[-1]["content"].insert(0, reflection_injects)
        
        #### done
        message.extend(history_prompt)

        if self.lm_config.provider == "sglang":
            message = _add_modality_key_for_sglang_messages(message)
        return message

    @time_it
    def on_task_end(self, trajectory: Trajectory, score: float, task_info: dict, meta_data: Any, search_tree_stats: None, **kwargs) -> None:
        assert search_tree_stats is not None, "search_tree_stats field is missing"
        Q = search_tree_stats["Q"]
        Nsa = search_tree_stats["Nsa"]
        P = search_tree_stats["P"]
        V_next = search_tree_stats["V_next"]

        # insert new task record and reflection
        mean_Q = np.mean([q for q in Q if q is not None])
        new_task_record = TaskRecord(
            trajectory=trajectory,
            task_info=task_info,
            Q=Q,
            Nsa=Nsa,
            P=P,
            V_next=V_next,
            final_score=score,
            est_final_score=1.0 if mean_Q > 0.7 else 0.0,
        )
        reflection_records = self.reflect(new_task_record)

        # save reflection record
        all_reflections: list[ReflectionRecord] = self._load_lzma_db_files(self.reflection_folder_path)
        logger.info(f"Found {len(all_reflections)} reflections from {self.reflection_folder_path}")
        existing_refl_hashes = set([hash(r) for r in all_reflections])
        new_refl_to_write = [r for r in reflection_records if hash(r) not in existing_refl_hashes]
        logger.info(f"Deduped and writing {len(new_refl_to_write)} new reflections to {self.reflection_folder_path}")
        self._write_lzma_db_files(self.reflection_folder_path, new_refl_to_write)

        # save task record in case we need to do analysis later
        all_task_records: list[TaskRecord] = self._load_lzma_db_files(self._task_record_folder_path)
        logger.info(f"Found {len(all_task_records)} task records from {self._task_record_folder_path}")
        existing_task_hashes = set([hash(r) for r in all_task_records])
        new_task_record_to_write = [r for r in [new_task_record] if hash(r) not in existing_task_hashes]
        logger.info(f"Deduped and writing {len(new_task_record_to_write)} new task records to {self._task_record_folder_path}")
        self._write_lzma_db_files(self._task_record_folder_path, new_task_record_to_write)
        return


class ReinforcedPolicyPConstructorPureTEXT(ReinforcedPolicyPConstructor):
    """
    + same as ReinforcedPolicyPConstructor, but runs in PURE TEXT MODALITY. This means:
    1. policy prompt contains no image
    2. reflection prompt contains no image
    (for simplicity, I directly implement this by MCoT istead of CoT prompt constructor)
    """
    is_multimodal = False

    def _is_long_history(self, all_prev_state_actions: Trajectory) -> bool:
        return len(all_prev_state_actions) >= 5 # (s,a,s,a,s)

    def _construct_context_specific_instruction(self, curr_task_intent: str, curr_obs: Observation) -> str:
        reflection_records = self.retrieve_reflections(curr_task_intent, curr_obs)
        self._retrieval_cache[(curr_task_intent, curr_obs["text"])] = reflection_records  # internally used by RMCTS agent

        reflection_texts = []
        for r_i, record in enumerate(reflection_records):
            record: ReflectionRecord

            intent = record.intent
            state_text = self.tokenizer.decode(self.tokenizer.encode(record.state_str)[:400])
            action_str = record.action_str
            next_state_text = self.tokenizer.decode(self.tokenizer.encode(record.next_state_str)[:200])
            reflection_text = self.tokenizer.decode(self.tokenizer.encode(record.reflection)[:128])
            
            reflection_texts.append((
                f"OBJECTIVE ({r_i+1}): {intent}\n"
                f"ATTEMPTED ACTION ({r_i+1}): {action_str}\n"
                f"REFLECTION ({r_i+1}): {reflection_text}"
            ))

        instruction = '\n#####\n'.join(reflection_texts).strip()
        logger.info(f"constructed context-specific instruction=\n{instruction}")
        return instruction

    def get_lm_api_input(
        self,
        intro: str,
        examples: list[tuple[str, str, str]],
        intent: str,
        all_prev_state_actions: Trajectory,
        all_prev_action_strs: list[str],
        context_specific_instruction: str,
    ):
        """Return the require format for an API"""
        assert self.lm_config.provider in ["openai", "sglang", "azure"], f"Provider {self.lm_config.provider} not implemented"
        assert self.lm_config.mode == "chat", f"Mode {self.lm_config.mode} not implemented"
        assert len(all_prev_state_actions) % 2 == 1, f"all_prev_state_actions should be in (s,a, ...,s,a,s), but got {len(all_prev_state_actions)=}"
        message: list[dict[str, str]] | str | list[str | Image.Image]

        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": intro}],
            }
        ]

        ### 1. add few shot examples
        if self._is_long_history(all_prev_state_actions):
            examples = []  # no need
        for (x, y) in examples:
            message.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": x},
                    ],
                }
            )
            message.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": y}],
                }
            )
        ### 2. add trajectory history
        ### 2.1 format init state
        history_prompt = []
        template = self.instruction["template"]
        init_template = self.instruction["init_template"]

        init_s = all_prev_state_actions[0]
        init_state_text = init_s["observation"]["text"]
        init_url = init_s["info"]["page"].url
        init_prev_action_str = all_prev_action_strs[0]
        hist_init = init_template.format(
            objective=intent,
            url=self.map_url_to_real(init_url),
            observation=init_state_text,
            previous_action=init_prev_action_str,
        )
        hist_init = (
            f"!IMPORTANT! Below is the task you need to solve. Please read the user's intent and input images (if any) carefully.\n"
            f"{hist_init}"
        )
        history_prompt.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": hist_init
                },
            ]
        })
        ### 2.2 format history actions
        past_action_idx = 0
        # skip first state. This should end with the last/current state
        for s_or_a in all_prev_state_actions[1:]:
            if isinstance(s_or_a, Action):
                history_prompt.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{s_or_a.raw_prediction}"
                            },
                        ]
                    }
                )
                past_action_idx += 1
            else:
                state_text = s_or_a["observation"]["text"]
                page = s_or_a["info"]["page"]
                url = page.url
                previous_action_str = all_prev_action_strs[past_action_idx]
                hist_current = template.format(
                    url=self.map_url_to_real(url),
                    observation=state_text,
                    previous_action=previous_action_str,
                )
                history_prompt.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": hist_current
                            },
                        ]
                    }
                )
        ### 2.3 inject reflection into current state
        additional_instructions = None
        reflection_injects = None
        if len(history_prompt) > 1:
            if context_specific_instruction != '':
                additional_instructions = (
                    "\nNOTE: Remember that you should consider user's intent, previous histories, "
                    "and reflections (if applicable) to better plan the next action."
                )
                reflection_injects = {
                    "type": "text",
                    "text": (
                        "REFLECTIONS: here are some relevant reflections from other tasks. "
                        "Note that these reflections may not be directly related to the new task below, but they may provide some useful insights.\n"
                        f"{context_specific_instruction}"
                    )
                }
            else:
                additional_instructions = (
                    "\nNOTE: Remember that you should consider user's intent and previous histories "
                    "to better plan the next action."
                )
        else:
            if context_specific_instruction != '':
                additional_instructions = (
                    "\nNOTE: Remember that you should consider user's intent and reflections (if applicable) "
                    "to better plan the next action."
                )
                reflection_injects = {
                    "type": "text",
                    "text": (
                        "REFLECTIONS: here are some relevant reflections from other tasks. "
                        "Note that these reflections may not be directly related to the new task below, but they may provide some useful insights.\n"
                        f"{context_specific_instruction}"
                    )
                }
        if additional_instructions is not None:
            history_prompt[-1]["content"][0]["text"] += additional_instructions
        if reflection_injects is not None:
            history_prompt[-1]["content"].insert(0, reflection_injects)
        
        #### done
        message.extend(history_prompt)
        return message

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ):
        # simply format prompt using the FULL trajectory
        intro = self.instruction["intro"]
        intro_w_reflections = self.instruction["intro_w_reflections"]
        intro_wo_icl = self.instruction["intro_wo_icl"]
        examples = self.instruction["examples"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        context_specific_instruction = self._construct_context_specific_instruction(
            curr_task_intent=intent, curr_obs=state_info["observation"],
        )
        if context_specific_instruction != "":
            intro = intro_w_reflections
        if self._is_long_history(trajectory):
            intro = intro_wo_icl
        
        ### format all past actions
        none_padded_action_history_str = copy.deepcopy(meta_data["action_history"])
        if none_padded_action_history_str[0].lower() != "none":
            none_padded_action_history_str.insert(0, "None")

        prompt = self.get_lm_api_input(
            intro, examples,
            intent=intent,
            all_prev_state_actions=trajectory,
            all_prev_action_strs=none_padded_action_history_str,
            context_specific_instruction=context_specific_instruction,
        )
        logger.info(f"constructed prompt with len={len(prompt)}")
        logger.debug(f"constructed full prompt:\n{display_multimodal_openai_messages(prompt)}")  # not really multimodal any more
        return prompt

    def _get_expectation_prompt(
        self,
        intro: str,
        curr_state_text: str,
        action_str: str,
    ):
        """Return the require format for an API"""
        message: list[dict[str, str]] | str | list[str | Image.Image]
        if self.rlm_config.provider in ["openai", "sglang", "azure"]:
            if self.rlm_config.mode == "chat":
                message = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": intro}],
                    }
                ]
                expectation_prompt = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": f"OBSERVATION:\n{curr_state_text}"
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Action: {action_str}",
                            },
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "What do you expect to happen after taking this action? "
                                    "Briefly describe what you think will appear on the webpage after performing the action."
                                )
                            },
                        ]
                    }
                ]
                message.extend(expectation_prompt)
                return message
        elif "google" in self.lm_config.provider:
            raise ValueError(
                f"Gemini models do not support yet"
            )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

    def _get_reflections_prompt(
        self,
        intro: str,
        curr_state_text: str,
        action_str: str,
        expecatation_str: str,
        next_state_text: str,
        metadata: dict
    ):
        task_success = metadata['success']
        task_status_str = "failed" if not task_success else "successfully completed"

        message: list[dict[str, str]] | list[str | Image.Image] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": intro}],
            }
        ]
        all_actions_before_current: list[Action] = metadata["all_actions_before_current"]
        if self.rlm_config.provider in ["openai", "sglang", "azure"]:
            if self.rlm_config.mode == "chat":
                user_start_content = [
                    {
                        "type": "text", 
                        "text": f"OBSERVATION:\n{curr_state_text}"
                    },
                    {
                        "type": "text", 
                        "text": f"OBJECTIVE: {metadata['intent']}"
                    },
                ]
                if len(all_actions_before_current) > 0:
                    formatted_action_history = "PREVIOUS ACTIONS taken by the agent:\n"
                    for a_idx, a in enumerate(all_actions_before_current):
                        formatted_action_history += f"({a_idx+1}) {a.raw_prediction}\n"
                    formatted_action_history = formatted_action_history.strip()
                    user_start_content.insert(
                        0, 
                        {
                            "type": "text",
                            "text": formatted_action_history,
                        }
                    )

                reflection_prompt = [
                    {
                        "role": "user",
                        "content": user_start_content
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Action: {action_str}\n\nExpectation for next observation: {expecatation_str}",
                            },
                        ]
                    }
                ]
                #### now we ask for reflection
                # next_state_img is None when we have a termination
                reflection_prompt.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": f"NEXT OBSERVATION:\n{next_state_text}"
                        },
                        {
                            "type": "text", 
                            "text": (
                                "Is this webpage what you expected? If not, can you conclude anything special about navigating on this website? "
                                "If you faced the same situation again, what would you do differently at a high level? Do NOT propose any specific actions/answers.\n"
                                "Keep your response within 100 words. "
                                f"Note that according to our evaluation, you have {task_status_str} the OBJECTIVE at the very end."
                            )
                        },
                    ]
                })
                message.extend(reflection_prompt)
                return message
        elif "google" in self.lm_config.provider:
            raise ValueError(
                f"Gemini models do not support yet"
            )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

    def _get_reflections(self, state: StateInfo, action: Action, next_state: StateInfo, metadata: dict) -> str:
        # prompt llm
        lm_config = self.rlm_config
        
        state_text = state["observation"]["text"]
        next_state_text = next_state["observation"]["text"]

        # format prompt
        expectation_prompt = self._get_expectation_prompt(
            intro=self.instruction["agent_intro"],
            curr_state_text=state_text,
            action_str=action.raw_prediction,
        )
        # this will use agent LLM API
        expecatation_ans = call_llm(
            lm_config,
            expectation_prompt,
            num_outputs=1
        )

        reflections_prompt = self._get_reflections_prompt(
            intro=self.instruction["agent_intro"],
            curr_state_text=state_text,
            action_str=action.raw_prediction,
            expecatation_str=expecatation_ans,
            next_state_text=next_state_text,
            metadata=metadata
        )
        gen_reflections = call_llm(
            lm_config,
            reflections_prompt,
            num_outputs=1
        )
        return gen_reflections
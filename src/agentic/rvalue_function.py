import os
import re
import copy
import lzma
import pickle
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from cachetools import Cache
from typing import Optional, Any
from PIL import Image
from langchain_openai import OpenAIEmbeddings
from langchain.schema.document import Document
from hashlib import sha256
from browser_env import Trajectory, ActionTypes
from browser_env.utils import StateInfo, pil_to_b64, Observation
from concurrent.futures import ThreadPoolExecutor
from src.logging import time_it
from src.llms import lm_config, call_llm
from src.llms.utils import _add_modality_key_for_sglang_messages
from src.llms.tokenizer import Tokenizer
from src.prompts.types import TaskRecord
from src.prompts.utils import FaissRetriever, display_multimodal_openai_messages
from src.agent.utils import _pil_image_to_str
from src.agentic.value_function import (
    CoTwRubricValueFunction, CoTwDebateValueFunction,
    VFUNC_DEBATE_INTRO,
    create_chat_completion_wrapper,
)
from src.envs.actions import Action
import hashlib
import logging


logger = logging.getLogger("rvalue_function")


class ValueReflectionRecord:
    pass


@dataclass
class RubricReflectionRecord(ValueReflectionRecord):
    # important!
    # retrieve given (intent, _intent_images_text, _init_image_text)
    intent: str  # 
    intent_images: Optional[list[Image.Image]]
    _intent_images_text: Optional[list[str]]  # for retrieval, captioned by rlm
    init_image: Image.Image
    _init_image_text: str  # for retrieval, captioned by rlm

    gen_rubric: str  # rubric generated from previous run
    action_history_so_far: list[Action]  # simplified representation of what happened so far
    curr_state_or_last_action: StateInfo | Action  # current state (or maybe the last stop action)
    estimated_V: float  # estimated value of the (actual) representation of what happened so far
    future_actions: list[Action]  # future actions

    reflection: str

    _task_success: bool  # whether the task was successful
    _from_task_hash: int  # map back to the task record

    def __hash__(self):
        # !! python's hash() is not deterministic
        encoded_image_str = _pil_image_to_str(self.intent_images)
        compressed_action_hist = "".join([a.raw_prediction for a in self.action_history_so_far])

        unique_str = self.intent + encoded_image_str + compressed_action_hist + str(self.estimated_V)
        hash_object = hashlib.md5(unique_str.encode())
        hash_int = int(hash_object.hexdigest(), 16)
        return hash_int

    def __eq__(self, other):
        if not isinstance(other, RubricReflectionRecord):
            return False
        return hash(self) == hash(other)

    def simplified_info(self) -> dict:
        return {
            "intent": self.intent,
            "reflection": self.reflection,
            "hash": hash(self),
        }


class ReinforcedValueFunctionMixin:
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

    def retrieve_reflections(
        self, *args, **kwargs
    ) -> list[ValueReflectionRecord]:
        raise NotImplementedError


R_VFUNC_INTRO_BASE = f"""
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not. If the current state is a failure but it looks like the agent is on the right track towards success, you should also output as such.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, the text in a comment or post, the date of a submission, etc. This may be formulated in the intent as "tell me", "what is", or "list out". The agent's last response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the agent encounters an exception and respond with the error content, the task is considered to be a failure. It is VERY IMPORTANT that the bot response is the stop action with the correct output. If the bot response is not stop (e.g., it is click, type, or goto), it is considered a failure for information seeking tasks.
2. Site navigation: The user wants to navigate to a specific page (which may also be specified in the intent as "find", "show me", "navigate to"). Carefully examine the agent's action history and the final state of the webpage (shown in the LAST IMAGE) to determine whether the agent successfully completes the task. It is VERY IMPORTANT that the agent actually navigates to the specified page (reflected by the final state of the webpage, in the LAST IMAGE) and NOT just output the name of the item or post. Make sure that the final url is compatible with the task. For example, if you are tasked to navigate to a comment or an item, the final page and url should be that of the specific comment/item and not the overall post or search page. If asked to navigate to a page with a similar image, make sure that an image on the page is semantically SIMILAR to the intent image. If asked to look for a particular post or item, make sure that the image on the page is EXACTLY the intent image. For this type of task to be considered successful, the LAST IMAGE and current URL should reflect the correct content. No need to consider the agent's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Ensure that the agent actually commits to the modification. For example, if the agent writes a review or a comment but does not click post, the task is considered to be a failure. Carefully examine the agent's action history and the final state of the webpage to determine whether the agent successfully completes the task. No need to consider the agent's response.

The <action>s the agent can perform fall into several categories:

Page Operation Actions:
```click [id]```: This action clicks on an element with a specific id on the webpage.
```type [id] [content] [1/0]```: Use this to type the content into the field with id, followed by pressing ``Enter`` to submit the form [1] or no submission [0].
```hover [id]```: Hover over an element with id.
```press [key_comb]```:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
```scroll [down]``` or ```scroll [up]```: Scroll the page up or down.

Tab Management Actions:
```new_tab```: Open a new, empty browser tab.
```tab_focus [tab_index]```: Switch the browser's focus to a specific tab using its index.
```close_tab```: Close the currently active tab.

URL Navigation Actions:
```goto [url]```: Navigate to a specific URL.
```go_back```: Navigate to the previously viewed page.
```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
```stop [answer/url]```: Issue this action when the task is complete. If the objective is to find a text-based answer (e.g., price), provide the answer in the bracket. If the objective is to find a link(s) to an item/post(s), provide the exact url(s) in the bracket (for example, stop [http://xxx]).

The agent should issue an <action> in the following format:
<reason for action>. In summary, the next action I will perform is ```<action>```.
""".strip()


R_VFUNC_INTRO = f"""
{R_VFUNC_INTRO_BASE}

Your task is to design a rubric to evaluate the agent's actions on a user's provided task. 
Below is the user's intent, and the last few state/action pairs of the agent's attempt to solve the provided task.
""".strip()


R_VFUNC_INTRO_W_REFLECTION = f"""
{R_VFUNC_INTRO_BASE}

Your task is to design a rubric to evaluate the agent's actions on a user's provided task. 
Below we provide a few relevant reflections about previous tasks, rubric designs and an agent's actual execution on that task.
This should help you design a better rubric to evaluate the agent's actions on the new task provided afterwards.
""".strip()


R_VFUNC_GEN_RUBRIC_PROMPT = """
Your task now is to provide a short rubric that can be used to evaluate if the user's intent has been successfully fulfilled.
The rubric should be based on the user's intent and the relevant images provided, and will be used to evaluate if the agent's final action has successfully completed the task.

To produce a good rubric, consider:
- What are some (simple) mistakes the agent could make that would lead to an incorrect answer?
- Any incorrect answer/action should fail at least one of the rubric points.
- Items in the rubric should be simple, clear, and easy to evaluate.
- You should have no more than four items in the rubric.

For example, if the user intent is to find a pink dress that costs less than $50, a good rubric could be:
Rubric:
[RUBRIC START]
1. Is the selected product a dress?
2. Is the selected dress pink?
3. Does the selected dress cost less than $50?
[RUBRIC END]
""".strip()


R_VFUNC_GEN_RUBRIC_PROMPT_W_REFLECTION = """
Your task now is to provide a short rubric that can be used to evaluate if the user's intent has been successfully fulfilled.
The rubric should be based on the user's intent and the relevant images provided, and will be used to evaluate if the agent's final action has successfully completed the task.

To produce a good rubric, consider:
- What are some (simple) mistakes the agent could make that would lead to an incorrect answer? You should refer to the reflections provided above to better understand the agent's capability.
- Any incorrect answer/action should fail at least one of the rubric points.
- Items in the rubric should be simple, clear, and easy to evaluate.
- You should have no more than four items in the rubric.

For example, if the user intent is to find a pink dress that costs less than $50, a good rubric could be:
Rubric:
[RUBRIC START]
1. Is the selected product a dress?
2. Is the selected dress pink?
3. Does the selected dress cost less than $50?
[RUBRIC END]
""".strip()


class ReinforcedCoTwRubricValueFunction(CoTwRubricValueFunction, ReinforcedValueFunctionMixin):
    def __init__(
        self,
        db_path : str | Path,
        rlm_config: lm_config.LMConfig,
        embedding_config: lm_config.LMConfig,
        rlm_tokenizer: Tokenizer,
        embedding_tokenizer: Tokenizer,
        ## behavioral args
        max_reflections_per_task: int = 2,
        reflection_threshold: float = 0.3,  # small errors will be corrected by MCTS itself
        min_retrieval_score: float = 0.25,
        max_to_retrieve: int = 1,  # vfunc has long context
        ## debugging
        use_gt_success: bool = False,
    ):
        # rlm is used to generate reflection, embedding is used for retrieval
        # which lm to use here as the final vfunc is controlled by environment variables (c.f. ReinforcedMCoTPromptConstructor)
        CoTwRubricValueFunction.__init__(self)
        ReinforcedValueFunctionMixin.__init__(
            self, db_path,
            embedding_config, embedding_tokenizer,
            rlm_config, rlm_tokenizer
        )
        self._image_caption_cache = Cache(maxsize=100)

        self.embedding_model_name = self.embedding_config.model
        self.max_reflections_per_task = max_reflections_per_task
        self.reflection_threshold = reflection_threshold
        self.min_retrieval_score = min_retrieval_score
        self.max_to_retrieve = max_to_retrieve
        self.use_gt_success = use_gt_success
        self.reflection_folder_path = os.path.join(self.db_path, "value_reflections")
        if not os.path.exists(self.reflection_folder_path):
            os.makedirs(self.reflection_folder_path, exist_ok=True)
        self.reflection_index_path = os.path.join(self.db_path, "value_reflections_index")
        self.all_reflections = []

        ## note that this path is the same as ReinforcedMCoTPromptConstructor
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

    def _caption_image(self, image: Image.Image) -> str:
        # call self.rlm to briefly describe the image
        encoded_image = pil_to_b64(image)
        if encoded_image in self._image_caption_cache:
            return self._image_caption_cache[encoded_image]
        
        prompts = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Below is a screenshot of a website augmented by set of marks annotations. "
                            "Please generate a short description about the items you see in the screenshot below. "
                            "Limit your response within 100 words."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_to_b64(image),
                        },
                    }
                ]
            },
        ]
        caption = create_chat_completion_wrapper(
            messages=prompts,
            model=self.rlm_config.model,
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            num_outputs=1
        )
        self._image_caption_cache[encoded_image] = caption
        return caption

    def _load_db(self):
        all_reflections: list[RubricReflectionRecord] = self._load_lzma_db_files(self.reflection_folder_path)
        logger.info(f"Loaded {len(all_reflections)} reflection records from {self.reflection_folder_path}")
        self.all_reflections = all_reflections

        ### embedding
        embedding_docs = []
        for i, record in enumerate(all_reflections):
            task_intent = record.intent
            task_intent_images = record.intent_images
            task_intent_images_text = record._intent_images_text
            init_screenshot = record.init_image
            init_screenshot_text = record._init_image_text

            # lets make it pure text first
            if len(task_intent_images_text) == 0:
                doc_text = f"Task: {task_intent}"
            else:
                task_image_texts = ""
                for i, img_text in enumerate(task_intent_images_text):
                    task_image_texts += f"Image {i} description: {img_text}\n"
                task_image_texts = task_image_texts.strip()
                doc_text = f"Task: {task_intent}\nTask Images:\n{task_image_texts}"
            
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

    @time_it
    def retrieve_reflections(
        self,
        curr_task_intent: str,
        curr_task_intent_images: Optional[list[Image.Image]],
        curr_task_intent_images_text: Optional[list[str]],
        init_screenshot: Image.Image,
        init_screenshot_text: str
    ) -> list[RubricReflectionRecord]:
        """Retrieve reflections from the database
        """
        curr_task_intent_images = curr_task_intent_images or []
        encoded_image_str = _pil_image_to_str(curr_task_intent_images + [init_screenshot])
        if (curr_task_intent, encoded_image_str) in self._retrieval_cache:
            logger.debug("fetching reflections from cache")
            return self._retrieval_cache[(curr_task_intent, encoded_image_str)]
        
        if self.retriever is None:
            logger.debug("Retriever not initialized. Skipping retrieve_reflections")
            return []
        
        # lets make it pure text first
        if len(curr_task_intent_images_text) == 0:
            query_str = f"Task: {curr_task_intent}"
        else:
            task_image_texts = ""
            for i, img_text in enumerate(curr_task_intent_images_text):
                task_image_texts += f"Image {i} description: {img_text}\n"
            task_image_texts = task_image_texts.strip()
            query_str = f"Task: {curr_task_intent}\nTask Images:\n{task_image_texts}"
        
        relevant_reflections = self.retriever.retrieve(
            query_str,
            min_score=self.min_retrieval_score,
            k=self.max_to_retrieve
        )
        reflection_records = []
        for doc in relevant_reflections:
            content_hashed = sha256(doc.page_content.encode()).hexdigest()
            record: RubricReflectionRecord = self._dochash_to_record[content_hashed]
            reflection_records.append(record)

        self._retrieval_cache[(curr_task_intent, encoded_image_str)] = reflection_records  # internally used by RMCTS agent
        return reflection_records

    def _construct_context_specific_instruction(
        self,
        curr_intent: str,
        curr_intent_images: list[Image.Image],
        curr_intent_images_text: list[str],
        curr_init_obs: Observation
    ) -> list[dict]:
        curr_init_screenshot = Image.fromarray(curr_init_obs["image"])
        reflection_records = self.retrieve_reflections(
            curr_intent,
            curr_intent_images,
            curr_intent_images_text,
            curr_init_screenshot,
            curr_init_obs["text"]
        )

        if len(reflection_records) == 0:
            logger.debug("skipping context-specific injection for value function due to zero retrieved results")
            return []
        logger.info(f"retrieved {len(reflection_records)} reflections for value function prompt injection")
        
        reflection_content = [
            {
                "type": "text",
                "text": (
                    "REFLECTIONS: here are some relevant reflections from other tasks. "
                    "Note that although these reflections may not be directly related to the current task at hand, "
                    "they may provide some useful insights."
                )
            }
        ]
        for vr_idx, vr_record in enumerate(reflection_records):
            vr_record: RubricReflectionRecord

            intent = vr_record.intent
            intent_images = vr_record.intent_images
            init_screenshot = vr_record.init_image
            rubric = vr_record.gen_rubric

            action_history_evaluated = vr_record.action_history_so_far
            if isinstance(vr_record.curr_state_or_last_action, Action):
                action_history_evaluated.append(vr_record.curr_state_or_last_action)
            action_history_evaluated_str = ""
            for a_idx, a in enumerate(action_history_evaluated):
                action_history_evaluated_str += f"Action {a_idx+1}: {a.raw_prediction}\n"
            action_history_evaluated_str = action_history_evaluated_str.strip()
            estimated_V = vr_record.estimated_V
            task_success = vr_record._task_success
            reflection = vr_record.reflection

            if task_success:
                success_str = "successfully completed"
                reflection_header_str = ""
            else:
                success_str = "failed"
                reflection_header_str = " How can you improve the rubric to better guide the agent towards success?"

            curr_reflection_content = [
                {
                    "type": "text",
                    "text": f"### REFLECTION {vr_idx+1}:",
                }
            ]
            if intent_images is not None:
                for i_img_idx, i_img in enumerate(intent_images):
                    curr_reflection_content.extend([
                        {
                            "type": "text",
                            "text": f"INTENT IMAGE: ({i_img_idx})",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": pil_to_b64(i_img),
                            },
                        }
                    ])
            curr_reflection_content.extend([
                {
                    "type": "text",
                    "text": (
                        f"User Intent: {intent}"
                    )
                },
                {
                    "type": "text",
                    "text": "IMAGES: (1) start page screenshot",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": pil_to_b64(init_screenshot)},
                },
            ])
            curr_reflection_content.append({
                "type": "text",
                "text": (
                    f"Rubric:\n{rubric}\n"
                    f"Note: an AI agent attempted this task under the guidance of this rubric, "
                    f"and has {success_str} to satisfy the user's original intent according to our evaluation."
                    f"{reflection_header_str}\n"
                    f"REFLECTION on the Rubric: {reflection}"
                )
            })
            reflection_content.extend(curr_reflection_content)

        injection = [
            {
                "role": "user",
                "content": reflection_content
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Thank you! I will take these reflections into account while designing a rubric "
                            "to evaluate the agent's actions on the new task provided below."
                        )
                    }
                ]
            }
        ]
        logger.info(f"constructed context-specific injection for value function={len(injection)}")
        return injection

    @staticmethod
    def _construct_rubric_prompt(
        intent: str,
        intent_images: list[Image.Image],
        init_screenshot: Image.Image,
        context_specific_instruction: list[dict] = []
    ) -> list:
        # init_screenshot is needed since sometimes the intent refers to "... on this page"
        start_content = []
        for i_img_idx, i_img in enumerate(intent_images):
            start_content.extend([
                {
                    "type": "text",
                    "text": f"INTENT IMAGE: ({i_img_idx})",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64(i_img),
                    },
                }
            ])
        start_content.extend([
            {
                "type": "text",
                "text": f"User Intent: {intent}"
            },
            {
                "type": "text",
                "text": "IMAGES: (1) start page screenshot",
            },
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(init_screenshot)},
            },
            {
                "type": "text",
                "text": "Rubric:\n"
            }
        ])
        if len(context_specific_instruction) == 0:
            messages = [
                {
                    "role": "system",
                    "content": R_VFUNC_INTRO
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": R_VFUNC_GEN_RUBRIC_PROMPT
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Got it! Please provide the user's intent or any relevant information I should use to create an evaluation rubric."
                        },
                    ]
                },
                {
                    "role": "user",
                    "content": start_content
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": R_VFUNC_INTRO_W_REFLECTION
                },
                *context_specific_instruction,
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": R_VFUNC_GEN_RUBRIC_PROMPT_W_REFLECTION
                        },
                        *start_content,  # model sucks at long context
                    ]
                }
            ]
        return messages
    
    @time_it
    def generate_rubrics(
        self,
        intent: str,
        intent_images: Optional[list[Image.Image]],
        init_screenshot: Image.Image,
        model: str
    ) -> str:
        # convert image to base64
        intent_images = intent_images or []
        encoded_image_str = _pil_image_to_str(intent_images + [init_screenshot])
        if (intent, encoded_image_str) in self._rubrics_cache:
            logger.debug("fetching rubric from cache")
            return self._rubrics_cache[(intent, encoded_image_str)]

        ### retrieve reflections, then generate
        logger.debug("generating new rubrics")
        curr_intent_images_text = []
        for i_img in intent_images:
            curr_intent_images_text.append(self._caption_image(i_img))
        init_screen_text = self._caption_image(init_screenshot)
        context_specific_instruction = self._construct_context_specific_instruction(
            curr_intent=intent,
            curr_intent_images=intent_images,
            curr_intent_images_text=curr_intent_images_text,
            curr_init_obs={
                "text": init_screen_text,
                "image": np.array(init_screenshot),
            }
        )
        messages = self._construct_rubric_prompt(
            intent,
            intent_images,
            init_screenshot,
            context_specific_instruction=context_specific_instruction,
        )
        response  = create_chat_completion_wrapper(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=256,
            top_p=0.9,
            context_length=-1,
            num_outputs=1,
        )
        try:
            # extracted_rubric = self._extract_rubric(response.choices[0].message.content)
            extracted_rubric = self._extract_rubric(response)
            # save to cache
            self._rubrics_cache[(intent, encoded_image_str)] = extracted_rubric
        except Exception as e:
            logger.error(e, exc_info=True)
            logger.error(f"Error extracting rubric")
            extracted_rubric = "None"
        return extracted_rubric
    
    def evaluate_success(
        self,
        screenshots: list[Image.Image],
        actions: list[str],
        current_url: str,
        last_reasoning: str,
        intent: str,
        models: list[str],
        init_screenshot: Optional[Image.Image] = None,
        intent_images: Optional[list[Image.Image]] = None,
        screenshots_text: Optional[list[str]] = None,
        n: int = 20, top_p: float = 1.0, should_log: bool = False
    ) -> float:
        # noop for now, as it automatically call our own generate_rubrics
        assert init_screenshot is not None, "init_screenshot is required for ReinforcedCoTwRubricValueFunction"
        success_score = super().evaluate_success(
            screenshots, actions, current_url, last_reasoning, intent,
            models, init_screenshot, intent_images, screenshots_text,
            n, top_p, should_log
        )
        return success_score

    def _get_expectation_prompt(
        self,
        past_actions: list[Action],
        past_states: list[StateInfo],
        curr_state_or_last_action: StateInfo | Action,
        metadata: dict
    ):
        """Return the require format for an API"""
        # past_actions + past_states should be (a,s,...,a) pairs
        # curr_state_or_last_action is what we want to evaluate
        intent = metadata["intent"]
        intent_images = metadata["intent_images"] or []
        init_screenshot = metadata["init_screenshot"]
        rubric = metadata["rubric"]

        assert self.rlm_config.provider in ["openai", "sglang", "azure"]
        assert self.rlm_config.mode == "chat"

        messages: list[dict[str, str]] | str | list[str | Image.Image]
        start_content = []
        for i_img_idx, i_img in enumerate(intent_images):
            start_content.extend([
                {
                    "type": "text",
                    "text": f"INTENT IMAGE: ({i_img_idx})",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64(i_img),
                    },
                }
            ])
        start_content.append({
            "type": "text",
            "text": (
                f"User Intent: {intent}"
            )
        })
        start_content.extend([
            {
                "type": "text",
                "text": "IMAGES: (1) start page screenshot",
            },
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(init_screenshot)},
            },
        ])
        messages = [
            {
                "role": "system",
                "content": R_VFUNC_INTRO_BASE
            },
            {
                "role": "user",
                "content": start_content
            },
        ]
        for i, (state, action) in enumerate(zip(past_states[1:], past_actions[:])):
            # action
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"Action: {action.raw_prediction}"
                    },
                ]
            })
            # next state
            state_screenshot = Image.fromarray(state["observation"]["image"])
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "IMAGES: (1) current page screenshot",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(state_screenshot)},
                    },
                ]
            })
        if isinstance(curr_state_or_last_action, Action):
            # last action of the task
            expectation_prompt = [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Action: {curr_state_or_last_action.raw_prediction}"
                        },
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": f"The agent's last action before task terminates:\n{curr_state_or_last_action.raw_prediction}"
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Rubric to evaluate the agent's performance:\n{rubric}\n\n"
                                "How well do you think the agent has performed according to this rubric?\n"
                                "Briefly explain why you think the agent has succeeded or failed."
                            )
                        }
                    ]
                }
            ]
        else:
            # recall that for normal states eval, len(past_states) == len(past_actions)
            prev_action = past_actions[-1]
            state_screenshot = Image.fromarray(curr_state_or_last_action["observation"]["image"])
            expectation_prompt = [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Action: {prev_action.raw_prediction}"
                        },
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"IMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(state_screenshot)},
                        },
                        {
                            "type": "text",
                            "text": (  # no need of rubric here
                                "What do you expect the agent to do next in order to fulfill the user's task?\n"
                                "Briefly describe what you think the agent is planning to do for the next few steps."
                            )
                        }
                    ]
                }
            ]
        messages.extend(expectation_prompt)

        if "sglang" == self.rlm_config.provider:
            messages = _add_modality_key_for_sglang_messages(messages)
        return messages

    def _get_reflections_prompt(
        self,
        past_actions: list[Action],
        past_states: list[StateInfo],
        curr_state_or_last_action: StateInfo | Action,
        expectation_str: str,
        future_actions: list[Action],
        final_score: float,
        metadata: dict
    ):
        task_success = final_score == 1.0
        task_status_str = "failed" if not task_success else "successfully completed"
        intent = metadata["intent"]
        intent_images = metadata["intent_images"]
        init_screenshot = metadata["init_screenshot"]
        rubric = metadata["rubric"]

        messages: list[dict[str, str]] | list[str | Image.Image] = []
        assert self.rlm_config.provider in ["openai", "sglang", "azure"]
        assert self.rlm_config.mode == "chat"

        start_content = []
        for i_img_idx, i_img in enumerate(intent_images):
            start_content.extend([
                {
                    "type": "text",
                    "text": f"INTENT IMAGE: ({i_img_idx})",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64(i_img),
                    },
                }
            ])
        start_content.append({
            "type": "text",
            "text": (
                f"User Intent: {intent}"
            )
        })
        start_content.extend([
            {
                "type": "text",
                "text": "IMAGES: (1) start page screenshot",
            },
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(init_screenshot)},
            },
        ])
        messages = [
            {
                "role": "system",
                "content": R_VFUNC_INTRO_BASE
            },
            {
                "role": "user",
                "content": start_content
            },
        ]
        for i, (state, action) in enumerate(zip(past_states[1:], past_actions[:])):
            # action
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"Action: {action.raw_prediction}"
                    },
                ]
            })
            # next state
            state_screenshot = Image.fromarray(state["observation"]["image"])
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "IMAGES: (1) current page screenshot",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(state_screenshot)},
                    },
                ]
            })
        #### now we ask for reflection
        task_hint = ""
        if not task_success:
            task_hint = (
                "This indicates the LAST ACTION did not correctly fulfill some aspects of "
                "the user's intent (e.g., color, price, material, provided intent image). "
            )
        if isinstance(curr_state_or_last_action, Action):
            # expectation for last action
            messages.extend([
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"LAST ACTION: {curr_state_or_last_action.raw_prediction}"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Thank you. Task is now terminated. Do you think the agent has successfully completed user's intent?"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"EVALUATION RUBRIC to access task progress:\n{rubric}\n\n"
                                f"EXPECTATION: {expectation_str}"
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": (
                                f"According to our evaluation of the LAST ACTION, the agent has {task_status_str} the OBJECTIVE. "
                                f"{task_hint}"
                                "Is the result same as what you expected? If not, what do you think went wrong?\n"
                                "If you were to design the evaluation rubric again to better guide this agent, what would you do differently at a high level?\n"
                                "Keep your response within 100 words."
                            )
                        }
                    ]
                }
            ])
        else:
            prev_action = past_actions[-1]
            state_screenshot = Image.fromarray(curr_state_or_last_action["observation"]["image"])
            messages.extend([
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Action: {prev_action.raw_prediction}"
                        },
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"IMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(state_screenshot)},
                        },
                        {
                            "type": "text",
                            "text": (  # no need of rubric here
                                "What do you expect the agent to do next in order to fulfill the user's task?\n"
                                "And how would you evaluate the agent's task progress at the end?"
                            )
                        }
                    ]
                }
            ])
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"EXPECTED FUTURE ACTIONS: {expectation_str}\n\n"
                            f"EVALUATION RUBRIC to access task progress:\n{rubric}"
                        )
                    }
                ]
            })
            all_future_action_str = ""
            for a_idx, n_action in enumerate(future_actions):
                all_future_action_str += f"Action {a_idx+1}. {n_action.raw_prediction}\n"
            all_future_action_str = all_future_action_str.strip()
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"AGENT'S ACTUAL FUTURE ACTIONS:\n{all_future_action_str}"
                    },
                    {
                        "type": "text", 
                        "text": (
                            "Is this what you expected the agent to do next? "
                            "If not, can you conclude anything noteworthy about this agent's overall capability at solving this task? "
                            f"Note that according to our evaluation of the last action/state, the agent has {task_status_str} the OBJECTIVE. "
                            f"{task_hint}"
                            "If you were to design the evaluation rubric again to better guide this agent, what would you do differently at a high level?\n"
                            "Keep your response within 100 words."
                        )
                    }
                ]
            })

        if "sglang" == self.rlm_config.provider:
            messages = _add_modality_key_for_sglang_messages(messages)
        return messages

    def _get_reflections(
        self,
        past_actions: list[Action],
        past_states: list[StateInfo],
        curr_state_or_last_action: StateInfo | Action,
        future_actions: list[Action],
        final_score: float,
        metadata: dict
    ) -> str:
        # prompt llm
        lm_config = self.rlm_config

        # format prompt
        expectation_prompt = self._get_expectation_prompt(
            past_actions=past_actions,
            past_states=past_states,
            curr_state_or_last_action=curr_state_or_last_action,
            metadata=metadata
        )
        # this will use agent LLM API
        expectation_ans = call_llm(
            lm_config,
            expectation_prompt,
            num_outputs=1
        )

        # understand more about the agent's capability and mistakes it tends to make
        # make suggestions about how to design rubric
        reflections_prompt = self._get_reflections_prompt(
            past_actions=past_actions,
            past_states=past_states,
            curr_state_or_last_action=curr_state_or_last_action,
            expectation_str=expectation_ans,
            future_actions=future_actions,
            final_score=final_score,
            metadata=metadata
        )
        gen_reflections = call_llm(
            lm_config,
            reflections_prompt,
            num_outputs=1
        )
        return gen_reflections

    def _selection_metric(self, task_record: TaskRecord) -> float:
        # ideally we want to compute V(s) - mean(Q(s,a)) for all a generated
        # here, for simplicity we compute V(s) - best Q(s,a)
        expected_v = task_record.V_next
        actual_Qsa = task_record.Q[1:]
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

    def _get_state_or_action_to_reflect(self, task_record: TaskRecord, used_idx: set):
        state_scores_ = self._selection_metric(task_record)
        most_unexpected_idx_sorted = np.argsort(state_scores_)[::-1]

        all_actions = []
        all_states = []
        for data in task_record.trajectory:
            if isinstance(data, Action):
                all_actions.append(data)
            else:
                all_states.append(data)

        found_idx = -1
        for i in most_unexpected_idx_sorted:
            if i not in used_idx:
                found_idx = i
                break
        assert found_idx != -1, "No more state to reflect on"

        trajectory = task_record.trajectory
        init_screenshot_array = trajectory[0]["observation"]["image"]
        init_screenshot = Image.fromarray(init_screenshot_array)

        ### find action/state to reflect on
        if (found_idx + 1) * 2 == len(trajectory):
            # last action
            last_action = trajectory[-1]
            assert isinstance(last_action, Action), "Last action is not an action"

            past_actions = all_actions[:found_idx]
            past_states = all_states[:found_idx+1]  # states = actions + 1

            curr_state_or_last_action = last_action

            future_actions = all_actions[found_idx+1:]  # should be empty
            final_score = task_record.final_score if self.use_gt_success else task_record.est_final_score

            V_estimated = task_record.V_next[found_idx]
        else:
            # a state
            state = trajectory[(found_idx + 1) * 2]  # recall that S_0 is never evaled
            assert isinstance(state, dict), "State is not a state"

            past_actions = all_actions[:found_idx+1]
            past_states = all_states[:found_idx+1]  # states = actions

            curr_state_or_last_action = state

            future_actions = all_actions[found_idx+1:]
            final_score = task_record.final_score if self.use_gt_success else task_record.est_final_score

            V_estimated = task_record.V_next[found_idx]
        metadata = {
            "intent": task_record.task_info["intent"],
            "intent_images": task_record.task_info["images"],
            "init_screenshot": init_screenshot,
            "V": V_estimated,
            "rubric": task_record._rubric
        }
        return (
            past_actions,
            past_states,
            curr_state_or_last_action,
            future_actions,
            final_score,
            metadata
        )

    @time_it
    def reflect(self, task_record: TaskRecord) -> list[RubricReflectionRecord]:
        # ask llm what it expect the agent to do, and what will happen if we let the agent cook
        # compare that with the actual trajectory and task status. gen reflection about agent's ability
        """Reflect on the task and return lessons learned"""
        max_num_records = self.max_reflections_per_task
        tmp_task_record = copy.deepcopy(task_record)
        # remove the last state so that its always (s,a,s,a...,s,last_a)
        if isinstance(tmp_task_record.trajectory[-1], dict):
            tmp_task_record.trajectory.pop(-1)
        # adjust Q, V from (q, v_next, q, v_next) to (q, v_next, q, v_next, final score) for _selection_metric
        used_idx = set()
        if self.use_gt_success:
            # RLHF type
            tmp_task_record.Q.append(tmp_task_record.final_score)
        else:
            # fully self-supervised
            tmp_task_record.Q.append(tmp_task_record.est_final_score)
        state_scores_ = self._selection_metric(tmp_task_record)
        most_unexpected_idx_sorted = np.argsort(state_scores_)[::-1]
        
        reflection_records = []
        for i in range(max_num_records):
            logger.info(f"Reflecting iteration {i}")
            # we are either evaluating a state (and its previous actions)
            # or the last action (since there is no next state)
            found_idx = -1
            for try_idx in most_unexpected_idx_sorted:
                if try_idx not in used_idx:
                    found_idx = try_idx
                    break
            if found_idx == -1:
                logger.info(f"No more state to reflect on, {used_idx=}")
                break
            most_unexpected_score = state_scores_[found_idx]

            if most_unexpected_score < self.reflection_threshold:
                logger.info(f"unexpected score {most_unexpected_score} is below {self.reflection_threshold=}. Stopping reflection.")
                break
            
            (
                past_actions,
                past_states ,
                curr_state_or_last_action,
                future_actions,
                final_score,
                metadata
            ) = self._get_state_or_action_to_reflect(tmp_task_record, used_idx)

            reflection = self._get_reflections(
                past_actions,
                past_states,
                curr_state_or_last_action,
                future_actions,
                final_score,
                metadata
            )
            logger.info(f"reinforced rubric value function generated reflection: {reflection}")

            ##### save reflection
            # first, caption the images which we will use as (text-based) retrieval
            # note that these caption calls should be cached (by calling generate_rubrics)
            intent_images_text = []
            intent_images = metadata["intent_images"]
            if intent_images is not None:
                for i_img in intent_images:
                    intent_images_text.append(self._caption_image(i_img))
            init_image = metadata["init_screenshot"]
            init_image_text = self._caption_image(init_image)

            ### clean some metadata to save space
            cleaned_past_actions = []
            for a_ in past_actions:
                a_copy: Action = copy.deepcopy(a_)
                a_copy.metadata.pop("obs_metadata", None)
                cleaned_past_actions.append(a_copy)
            new_reflection_record = RubricReflectionRecord(
                intent=metadata["intent"],
                intent_images=metadata["intent_images"],
                _intent_images_text=intent_images_text,
                init_image=init_image,
                _init_image_text=init_image_text,
                # other stuff
                gen_rubric=metadata["rubric"],
                action_history_so_far=cleaned_past_actions,
                curr_state_or_last_action=curr_state_or_last_action,
                estimated_V=metadata["V"],
                future_actions=future_actions,
                reflection=reflection,
                _task_success=final_score == 1.0,
                _from_task_hash=hash(task_record)
            )
            reflection_records.append(new_reflection_record)

            # update to prepare for next iteration
            used_idx.add(found_idx)
        return reflection_records

    @time_it
    def on_task_end(self, trajectory: Trajectory, score: float, task_info: dict, meta_data: Any, search_tree_stats: None, **kwargs) -> None:
        assert search_tree_stats is not None, "search_tree_stats field is missing"
        Q = search_tree_stats["Q"]
        Nsa = search_tree_stats["Nsa"]
        P = search_tree_stats["P"]
        V_next = search_tree_stats["V_next"]

        # this should already be cached
        init_screenshot_array = trajectory[0]["observation"]["image"]
        init_screenshot = Image.fromarray(init_screenshot_array)
        intent_images = task_info['images']
        intent = task_info['intent']
        encoded_image_str = _pil_image_to_str(intent_images + [init_screenshot])
        rubric = self._rubrics_cache.get((intent, encoded_image_str), "")

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
            _rubric=rubric
        )
        reflection_records = self.reflect(new_task_record)

        # save reflection record
        all_reflections: list[RubricReflectionRecord] = self._load_lzma_db_files(self.reflection_folder_path)
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



@dataclass
class DebateReflectionRecord(ValueReflectionRecord):
    # important!
    # retrieve given (intent, _intent_images_text, curr_state_or_last_action)
    intent: str  # 
    intent_images: Optional[list[Image.Image]]
    _intent_images_text: Optional[list[str]]  # for retrieval, captioned by rlm
    partial_trajectory_to_eval: Trajectory  # current state (or maybe the last stop action)

    gen_supporting_reasons: str  # supporting reasons generated from previous run
    gen_opposing_reasons: str  # opposing reasons generated from previous run
    estimated_V: float  # estimated value of the (actual) representation of what happened so far

    reflection: str

    _task_success: bool  # whether the task was successful
    _from_task_hash: int  # map back to the task record

    def __hash__(self):
        # !! python's hash() is not deterministic
        encoded_image_str = _pil_image_to_str(self.intent_images)

        init_state = self.partial_trajectory_to_eval[0]
        init_state_img_array = init_state["observation"]["image"]
        init_state_img = Image.fromarray(init_state_img_array)
        init_state_img_encoded = _pil_image_to_str([init_state_img])
        traj_str = ""
        for data in self.partial_trajectory_to_eval[1:]:
            if isinstance(data, Action):
                traj_str += f" -> {data.raw_prediction}"
        
        unique_str = self.intent + encoded_image_str + init_state_img_encoded + traj_str
        hash_object = hashlib.md5(unique_str.encode())
        hash_int = int(hash_object.hexdigest(), 16)
        return hash_int

    def __eq__(self, other):
        if not isinstance(other, DebateReflectionRecord):
            return False
        return hash(self) == hash(other)

    def simplified_info(self) -> dict:
        return {
            "intent": self.intent,
            "reflection": self.reflection,
            "hash": hash(self),
        }


class ReinforcedDebateValueFunction(CoTwDebateValueFunction, ReinforcedValueFunctionMixin):
    # same as CoTwDebateValueFunction. A dummy class for RMCTS agent
    def __init__(
        self,
        db_path : str | Path,
        rlm_config: lm_config.LMConfig,
        embedding_config: lm_config.LMConfig,
        rlm_tokenizer: Tokenizer,
        embedding_tokenizer: Tokenizer,
        ## behavioral args
        max_reflections_per_task: int = 2,
        reflection_threshold: float = 0.3,  # small errors will be corrected by MCTS itself
        min_retrieval_score: float = 0.25,
        max_to_retrieve: int = 1,  # vfunc has long context
        ## debugging
        use_gt_success: bool = False,
    ):
        # rlm is used to generate reflection, embedding is used for retrieval
        # which lm to use here as the final vfunc is controlled by environment variables (c.f. ReinforcedMCoTPromptConstructor)
        CoTwDebateValueFunction.__init__(self)
        ReinforcedValueFunctionMixin.__init__(
            self, db_path,
            embedding_config, embedding_tokenizer,
            rlm_config, rlm_tokenizer
        )
        self._image_caption_cache = Cache(maxsize=100)

        self.embedding_model_name = self.embedding_config.model
        self.max_reflections_per_task = max_reflections_per_task
        self.reflection_threshold = reflection_threshold
        self.min_retrieval_score = min_retrieval_score
        self.max_to_retrieve = max_to_retrieve
        self.use_gt_success = use_gt_success
        
        self.reflection_folder_path = os.path.join(self.db_path, "value_reflections")
        if not os.path.exists(self.reflection_folder_path):
            os.makedirs(self.reflection_folder_path, exist_ok=True)
        self.reflection_index_path = os.path.join(self.db_path, "value_reflections_index")
        
        self.all_reflections = []  # judge reflections

        ## note that this path is the same as ReinforcedMCoTPromptConstructor
        self._task_record_folder_path = os.path.join(self.db_path, "task_records")
        if not os.path.exists(self._task_record_folder_path):
            os.makedirs(self._task_record_folder_path, exist_ok=True)
        self._all_task_records = []
        self._dochash_to_record = {}
        self.retriever: FaissRetriever = None
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

    def _caption_image(self, image: Image.Image) -> str:
        # call self.rlm to briefly describe the image
        encoded_image = pil_to_b64(image)
        if encoded_image in self._image_caption_cache:
            return self._image_caption_cache[encoded_image]
        
        prompts = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Below is a screenshot of a website augmented by set of marks annotations. "
                            "Please generate a short description about the items you see in the screenshot below. "
                            "Limit your response within 100 words."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_to_b64(image),
                        },
                    }
                ]
            },
        ]
        caption  = create_chat_completion_wrapper(
            messages=prompts,
            model=self.rlm_config.model,
            temperature=0.7,
            max_tokens=256,
            top_p=0.9,
            num_outputs=1
        )
        self._image_caption_cache[encoded_image] = caption
        return caption

    def _load_db(self):
        all_reflections: list[DebateReflectionRecord] = self._load_lzma_db_files(self.reflection_folder_path)
        logger.info(f"Loaded {len(all_reflections)} judge reflection records from {self.reflection_folder_path}")
        self.all_reflections = all_reflections

        ### embedding
        embedding_docs = []
        for i, record in enumerate(all_reflections):
            task_intent = record.intent
            task_intent_images = record.intent_images
            task_intent_images_text = record._intent_images_text

            last_s_or_sa_str = ""
            last_state_or_action = record.partial_trajectory_to_eval[-1]
            if isinstance(last_state_or_action, Action):
                # embed state and last action
                second_last_state = record.partial_trajectory_to_eval[-2]
                last_s_or_sa_str = "State:\n" + second_last_state["observation"]["text"] + "Action:\n" + last_state_or_action.raw_prediction
            else:
                # just save the state
                last_s_or_sa_str = "State:\n" + last_state_or_action["observation"]["text"]

            # lets make it pure text first
            if len(task_intent_images_text) == 0:
                doc_text = f"Task: {task_intent}\n" + last_s_or_sa_str
            else:
                task_image_texts = ""
                for i, img_text in enumerate(task_intent_images_text):
                    task_image_texts += f"Image {i} description: {img_text}\n"
                task_image_texts = task_image_texts.strip()
                doc_text = f"Task: {task_intent}\nTask Images:\n{task_image_texts}\n" + last_s_or_sa_str
            
            doc = Document(doc_text)
            embedding_docs.append(doc)

            doc_text_hashed = sha256(doc_text.encode()).hexdigest()
            self._dochash_to_record[doc_text_hashed] = record
        
        logger.debug("Initializing FaissRetriever")
        if len(embedding_docs) > 0:
            retriever = FaissRetriever(
                index_save_path=self.reflection_index_path,
                docs=embedding_docs,
                embeddings=OpenAIEmbeddings(
                    model=self.embedding_model_name,
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                    organization=os.environ.get("OPENAI_ORGANIZATION", ""),
                )
            )
        else:
            retriever = None
        return retriever

    def on_task_start(self, task_info: dict, **kwargs) -> None:
        self.retriever = self._load_db()
        return

    @time_it
    def retrieve_reflections(
        self,
        curr_task_intent: str,
        curr_task_intent_images: Optional[list[Image.Image]],
        curr_task_intent_images_text: Optional[list[str]],
        current_state_image_to_eval: Image.Image,
        current_state_text_to_eval: str,
        last_action_text: str = ""
    ) -> list[RubricReflectionRecord]:
        """Retrieve reflections from the database to reinforce the judge.
        last_action_text is not empty if we are at the last state
        """

        curr_task_intent_images = curr_task_intent_images or []
        encoded_image_str = _pil_image_to_str(curr_task_intent_images + [current_state_image_to_eval])
        encoded_retrieval_key = (curr_task_intent, encoded_image_str, last_action_text)
        if encoded_retrieval_key in self._retrieval_cache:
            logger.debug("fetching reflections from cache")
            return self._retrieval_cache[encoded_retrieval_key]
        
        retriever = self.retriever
        if retriever is None:
            logger.debug("Retriever not initialized. Skipping retrieve_reflections")
            return []
        
        # lets make it pure text first
        last_s_or_sa_str = ""
        if last_action_text != "":
            # embed state and last action
            last_s_or_sa_str = "State:\n" + current_state_text_to_eval + "Action:\n" + last_action_text
        else:
            # just save the state
            last_s_or_sa_str = "State:\n" + current_state_text_to_eval
        
        if len(curr_task_intent_images_text) == 0:
            query_str = f"Task: {curr_task_intent}\n" + last_s_or_sa_str
        else:
            task_image_texts = ""
            for i, img_text in enumerate(curr_task_intent_images_text):
                task_image_texts += f"Image {i} description: {img_text}\n"
            task_image_texts = task_image_texts.strip()
            query_str = f"Task: {curr_task_intent}\nTask Images:\n{task_image_texts}\n" + last_s_or_sa_str

        relevant_reflections = retriever.retrieve(
            query_str,
            min_score=self.min_retrieval_score,
            k=self.max_to_retrieve
        )
        reflection_records = []
        for doc in relevant_reflections:
            content_hashed = sha256(doc.page_content.encode()).hexdigest()
            record: DebateReflectionRecord = self._dochash_to_record[content_hashed]
            reflection_records.append(record)

        self._retrieval_cache[encoded_retrieval_key] = reflection_records  # internally used by RMCTS agent
        return reflection_records

    def _construct_context_specific_instruction(
        self,
        curr_intent: str,
        curr_intent_images: Optional[list[Image.Image]],
        curr_intent_images_text: Optional[list[str]],
        curr_state_image: Image.Image,
        curr_state_text: str,
        last_action_desc: str = ""
    ):
        # reinforce the judge
        # last_action not none when its the last action (which we computed V on)
        if last_action_desc == "":
            reflection_records = self.retrieve_reflections(
                curr_intent,
                curr_intent_images,
                curr_intent_images_text,
                curr_state_image,
                curr_state_text,
                last_action_text=""
            )
        else:
            reflection_records = self.retrieve_reflections(
                curr_intent,
                curr_intent_images,
                curr_intent_images_text,
                curr_state_image,
                curr_state_text,
                last_action_text=last_action_desc
            )

        if len(reflection_records) == 0:
            logger.debug("skipping context-specific injection for value function due to zero retrieved results")
            return ""
        logger.info(f"retrieved {len(reflection_records)} reflections for value function prompt injection")
        
        reflection_texts = []
        for vr_idx, vr_record in enumerate(reflection_records):
            vr_record: DebateReflectionRecord

            intent = vr_record.intent
            intent_images = vr_record.intent_images
            
            gen_supporting_reasons = vr_record.gen_supporting_reasons
            gen_opposing_reasons = vr_record.gen_opposing_reasons
            task_success = vr_record._task_success
            reflection = vr_record.reflection

            reflection_texts.append((
                f"OBJECTIVE ({vr_idx+1}): {intent}\n"
                f"<agent's action omitted> <supporting reasons omitted> <opposing reasons omitted>\n"
                f"REFLECTION ({vr_idx+1}): {reflection}"
            ))
        instruction = '\n#####\n'.join(reflection_texts).strip()
        logger.info(f"constructed context-specific instruction for judge with\n{instruction=}")
        return instruction

    def _construct_final_decision_prompt(
        self,
        screenshots: list[Image.Image],  # all screenshots after the initial one
        screenshots_text: list[str],
        actions: list[str],
        last_state_url: str,
        last_reasoning: str,
        intent: str,
        intent_images: list[Image.Image],
        opposing_opinions: str,
        supporting_opinions: str,
        opposing_first: bool,
        context_specific_instruction: str,
    ) -> list:
        logger.debug(f'evaluating last response={actions[-1]}, reasoning={last_reasoning}')
        logger.debug(f"received {len(screenshots)=} and {len(actions)=}")
        logger.debug(f"using context_specific_instruction with {len(context_specific_instruction)=}")
        messages = super()._construct_final_decision_prompt(
            screenshots, screenshots_text, actions, last_state_url, last_reasoning, intent, intent_images,
            opposing_opinions, supporting_opinions, opposing_first
        )

        if context_specific_instruction != '':
            line_to_edit = "Format your response into two lines as shown below. Keep your response concise."
            assert line_to_edit in messages[-1]["content"][-1]["text"]

            additional_instructions = (
                " You should consider the reasons above and the provided reflections (if applicable) to come up with the most accurate judgement."
            )
            reflection_injects = {
                "type": "text",
                "text": (
                    "REFLECTIONS: here are some relevant reflections from other tasks. "
                    "Note that these reflections may NOT be directly related to the current task/agent's performance, "
                    "but they may provide some useful insights.\n"
                    f"{context_specific_instruction}"
                )
            }
            messages[-1]["content"][-1]["text"] = messages[-1]["content"][-1]["text"].replace(
                line_to_edit,
                line_to_edit + additional_instructions
            )
            messages[-1]["content"].insert(-1, reflection_injects)  # goes to the second last position
        return messages

    def generate_final_decisions(
        self,
        screenshots: list[Image.Image],
        screenshots_text: list[str],
        actions: list[str],
        last_state_url: str,
        last_reasoning: str,
        intent: str,
        intent_images: list[Image.Image],
        opposing_opinions: str,
        supporting_opinions: str,
        model: str
    ) -> list[str]:
        ### generate
        logger.debug("generating supporting arguments")

        curr_screenshot = screenshots[-1]
        curr_screenshot_text = screenshots_text[-1]
        intent_images_text = []
        if intent_images is not None:
            for i_img in intent_images:
                intent_images_text.append(self._caption_image(i_img))

        if len(actions) == len(screenshots):
            # (s,a,s,a) situation
            last_action_desc = actions[-1]
        else:
            # (s,a,s) situation
            last_action_desc = ""
        context_specific_instruction = self._construct_context_specific_instruction(
            curr_intent=intent,
            curr_intent_images=intent_images,
            curr_intent_images_text=intent_images_text,
            curr_state_image=curr_screenshot,
            curr_state_text=curr_screenshot_text,
            last_action_desc=last_action_desc
        )

        ## prompts
        oppose_first_messages = self._construct_final_decision_prompt(
            screenshots=copy.deepcopy(screenshots),
            screenshots_text=copy.deepcopy(screenshots_text),
            actions=copy.deepcopy(actions),
            last_state_url=last_state_url,
            last_reasoning=last_reasoning,
            intent=intent,
            intent_images=intent_images,
            opposing_opinions=opposing_opinions,
            supporting_opinions=supporting_opinions,
            opposing_first=True,
            context_specific_instruction=context_specific_instruction
        )
        support_first_messages = self._construct_final_decision_prompt(
            screenshots=copy.deepcopy(screenshots),
            screenshots_text=copy.deepcopy(screenshots_text),
            actions=copy.deepcopy(actions),
            last_state_url=last_state_url,
            last_reasoning=last_reasoning,
            intent=intent,
            intent_images=intent_images,
            opposing_opinions=opposing_opinions,
            supporting_opinions=supporting_opinions,
            opposing_first=False,
            context_specific_instruction=context_specific_instruction
        )
        logger.debug(f"vfunc constructed final decision prompt 1:\n{display_multimodal_openai_messages(oppose_first_messages)}")
        logger.debug(f"vfunc constructed final decision prompt 2: omitted")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            futures.append(executor.submit(
                create_chat_completion_wrapper,
                model=model,
                messages=oppose_first_messages,
                max_tokens=256,
                temperature=1.0,
                top_p=0.95,
                num_outputs=3
            ))
            futures.append(executor.submit(
                create_chat_completion_wrapper,
                model=model,
                messages=support_first_messages,
                max_tokens=256,
                temperature=1.0,
                top_p=0.95,
                num_outputs=3
            ))

            oppose_first_response, support_first_response = [f.result() for f in futures]

        # oppose_first_judge_ans = [r.message.content for r in oppose_first_response.choices]
        oppose_first_judge_ans = oppose_first_response
        oppose_first_judge_ans_str = '\n'.join(oppose_first_judge_ans)
        logger.debug(f"Oppose first judge answers:\n{oppose_first_judge_ans_str}")

        # support_first_judge_ans = [r.message.content for r in support_first_response.choices]
        support_first_judge_ans = support_first_response
        support_first_judge_ans_str = '\n'.join(support_first_judge_ans)
        logger.debug(f"Support first judge answers:\n{support_first_judge_ans_str}")
        return oppose_first_judge_ans + support_first_judge_ans
    
    # changed value range to (-1,1)
    @time_it
    def evaluate_success(
        self,
        screenshots: list[Image.Image],
        actions: list[str],
        current_url: str,
        last_reasoning: str,
        intent: str,
        models: list[str],
        init_screenshot: Optional[Image.Image] = None,
        intent_images: Optional[list[Image.Image]] = None,
        screenshots_text: Optional[list[str]] = None,
        n: int = 20, top_p: float = 1.0, should_log: bool = False
    ) -> float:
        """Compute the value of a state using the value function.
        """
        assert screenshots_text is not None, "Need text for each screenshot in reinforced debate value function"
        cache_key = self._encode_eval_success_input(
            screenshots=screenshots,
            actions=actions,
            intent=intent,
            intent_images=intent_images or []
        )
        if cache_key in self._debate_cache:
            data_dict = self._debate_cache[cache_key]
            return data_dict['v']
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            futures.append(executor.submit(
                self.generate_supporting_opinions,
                copy.deepcopy(screenshots),  # histories
                copy.deepcopy(screenshots_text),  # histories
                copy.deepcopy(actions),  # histories
                current_url,
                last_reasoning,
                intent,
                intent_images,
                models[0]
            ))
            futures.append(executor.submit(
                self.generate_opposing_opinions,
                copy.deepcopy(screenshots),  # histories
                copy.deepcopy(screenshots_text),  # histories
                copy.deepcopy(actions),  # histories
                current_url,
                last_reasoning,
                intent,
                intent_images,
                models[0]
            ))

            supporting_opinions, opposing_opinions = [f.result() for f in futures]
            logger.info(f"Value function generated supporing opinions:\n{supporting_opinions}")
            logger.info(f"Value function generated opposing opinions:\n{opposing_opinions}")
        
        final_decisions = self.generate_final_decisions(
            screenshots=screenshots,  # will be copied in the function
            screenshots_text=screenshots_text,
            actions=actions,  # will be copied in the function
            last_state_url=current_url,
            last_reasoning=last_reasoning,
            intent=intent,
            intent_images=intent_images,
            opposing_opinions=opposing_opinions,
            supporting_opinions=supporting_opinions,
            model=models[0]
        )

        ### parse the responses to scores
        all_scores = []
        for r_idx, resp_text in enumerate(final_decisions):
            logger.debug(f"Final decision {r_idx}: {resp_text}")
            try:
                pred = re.search(r'.*STATUS CODE: (\w).*', resp_text).group(1)
                if 'A' in pred:
                    score = 1.0
                elif 'B' in pred:
                    score = 0.7
                elif 'C' in pred:
                    score = 0.5
                elif 'D' in pred:
                    score = -0.2
                else:
                    score = -1.0
            except Exception as e:
                print(f"Error parsing response: {e}")
                score = 0.0
            
            all_scores.append(score)
        score = np.mean(all_scores)
        logger.debug(f"Final score from vfunc {models}: {score} from {all_scores}")
        
        ### save this to cache
        if cache_key not in self._debate_cache:
            self._debate_cache[cache_key] = {
                'v': score,
                'supporting_reasons': supporting_opinions,
                'opposing_reasons': opposing_opinions,
                'final_decisions': final_decisions
            }
        return score

    def _get_improve_judge_prompt(
        self,
        intent: str,
        intent_images: Optional[list[Image.Image]],
        trajectory: Trajectory,
        task_success: bool,
        metadata: dict,
    ):
        last_state_url = ""
        screenshots = []
        actions = []
        gen_debate_reasons = metadata["debate"]

        for data in trajectory:
            if isinstance(data, Action):
                actions.append(data.raw_prediction)
            else:
                screenshots.append(Image.fromarray(data["observation"]["image"]))
                last_state_url = data["url"]
        logger.debug(f'constructing reinforced debate prompt for last response={actions[-1]}')
        logger.debug(f"received {len(screenshots)=} and {len(actions)=}")

        #### code copied over from CoTwDebateValueFunction
        max_turns = 4  # in the prompt, there will be max_turns + 1 screenshots and actions
        ## prepare screenshots and actions
        # trajectory is either (s,a,s,a,s), or (s,a,s,a) or stop actions
        if len(screenshots) == 1 and len(actions) == 1:
            # (s, stop)
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = None
            last_action = actions[0]

            last_k_screenshots = []
            last_k_actions = []

            has_truncation = False
        elif len(screenshots) > len(actions):
            # the first case (s,a,s,...,a,s)
            assert len(screenshots) >= 2
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = screenshots.pop(-1)
            last_action = None

            last_k_screenshots = screenshots[1:]
            last_k_screenshots = last_k_screenshots[-max_turns:]

            last_k_actions = actions[1:]
            last_k_actions = last_k_actions[-max_turns:]

            has_truncation = len(last_k_screenshots) + 2 < len(screenshots)
        else:
            # the second case (s,a,s,...,a)
            assert len(screenshots) >= 2
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = screenshots.pop(-1)
            last_action = actions.pop(-1)

            last_k_screenshots = screenshots[1:]
            last_k_screenshots = last_k_screenshots[-max_turns:]

            last_k_actions = actions[1:]
            last_k_actions = last_k_actions[-max_turns:]

            has_truncation = len(last_k_screenshots) + 2 < len(screenshots)

        ### 1. start content and init screenshot
        start_content = [
            {
                "type": "text",
                "text": (
                    f"User Intent: {intent}"
                )
            }
        ]
        i_img_idx = -1
        if intent_images is not None:
            intent_images_content = []
            for i_img_idx, i_img in enumerate(intent_images):
                intent_images_content.extend([
                    {
                        "type": "text",
                        "text": f"INTENT IMAGE: ({i_img_idx+1})",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_to_b64(i_img),
                        },
                    }
                ])
            start_content.extend(intent_images_content)
        start_content.extend([
            {
                "type": "text",
                "text": f"IMAGES: ({i_img_idx+2}) start page screenshot",
            },
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(init_screenshot)},
            }
        ])
        
        #### 2. first action. Now we have (s0,a0)
        messages = [
            {
                "role": "system",
                "content": VFUNC_DEBATE_INTRO
            },
            {
                "role": "user",
                "content": start_content
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": init_action
                    },
                ]
            }
        ]

        ##### 3. the rest of (s, a, ...) after the first one
        ## 3.1 loop enter all turns before the last turn
        for i, (screenshot, action) in enumerate(zip(last_k_screenshots, last_k_actions)):
            if i == 0 and has_truncation:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "(fast forwarding to the last few state/action pairs...)\nIMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(screenshot)},
                        },
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "IMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(screenshot)},
                        },
                    ]
                })
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": action
                    },
                ]
            })
        ## 3.2 last turn prompt, which could be either (s), (s,a), or nothing
        if last_screenshot is None:
            # the (s, stop) case
            # there is nothing
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Agent's final action was: {last_action}"
                    }
                ]
            })
        elif last_action is not None:
            # eval stop actions
            # there is a (s,a) pair
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Last page URL={last_state_url}\nIMAGES: (1) last page screenshot.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(last_screenshot)},
                    },
                    {
                        "type": "text",
                        "text": f"Agent's final action: {last_action}"
                    }
                ]
            })
        else:
            # there is a (s) case
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Last page URL={last_state_url}\nIMAGES: (1) last page screenshot.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(last_screenshot)},
                    }
                ]
            })

        ### 4. construct reflection prompt to improve supporting/opposing reasons
        task_status_str = "failed" if not task_success else "successfully completed"
        if not task_success:
            task_hint = (
                "This indicates that the agent failed to fulfill some aspects of "
                "the user's intent (e.g., wrong color, price, material, or is different from the provided intent image). "
            )
        else:
            task_hint = (
                "This indicates that the agent eventually managed to fulfill all requirements in the user's intent. "
            )

        opposing_reasons = gen_debate_reasons["opposing_reasons"]
        supporting_reasons = gen_debate_reasons["supporting_reasons"]
        if task_success:
            # boost judge's confidence on supporting reasons
            messages.append({
                'role': 'assistant',
                'content': [{
                    "type": "text",
                    "text": (
                        f"Evaluation:\n"
                        f"(SUPPORT) Reasons why the agent is on the right track: {supporting_reasons}\n"
                        f"(OPPOSE) Reasons why the agent is NOT on the right track: {opposing_reasons}\n"
                        f"Final judgment: According to the given user's intent, the agent's executed actions so far, and the reasons stated above, "
                        "I think the agent is likely NOT making a good progress."  # flip the judgment
                    )
                }]
            })
        else:
            # boost judge's confidence on opposing reasons
            messages.append({
                'role': 'assistant',
                'content': [{
                    "type": "text",
                    "text": (
                        f"Evaluation:\n"
                        f"(SUPPORT) Reasons why the agent is on the right track: {supporting_reasons}\n"
                        f"(OPPOSE) Reasons why the agent is NOT on the right track: {opposing_reasons}\n"
                        f"Final judgment: According to the given user's intent, the agent's executed actions so far, and the reasons stated above, "
                        "I think the agent is likely making a good progress."  # flip the judgment
                    )
                }]
            })
        messages.append({
            'role': 'user',
            'content': [{
                "type": "text",
                "text": (
                    f"According to our evaluation, the agent has {task_status_str} the OBJECTIVE when the task ended. "
                    f"{task_hint}"
                    "Is any of the provided reasons above (SUPPORT or OPPOSE) INCORRECT given the user's intent and the executed actions so far? If so, why? "
                    "Based on the provided reasons and the agent's actual executions, what advice would you give the judge to make a better final judgment next time?\n"
                    "Keep your response within 100 words."
                )
            }]
        })
        return messages

    def _get_reflections(
        self,
        partial_trajectory: Trajectory,
        final_score: float,
        metadata: dict,
    ):
        # prompt llm
        lm_config = self.rlm_config

        task_success = final_score == 1.0
        intent = metadata["intent"]
        intent_images = metadata["intent_images"]
        improve_prompt = self._get_improve_judge_prompt(
            intent=intent,
            intent_images=intent_images,
            trajectory=partial_trajectory,
            task_success=task_success,
            metadata=metadata
        )
        logger.debug(f"improve judge prompt:\n{display_multimodal_openai_messages(improve_prompt)}")
        
        gen_reflections = call_llm(
            lm_config,
            improve_prompt,
            num_outputs=1
        )
        return gen_reflections

    def _selection_metric(self, task_record: TaskRecord) -> float:
        # ideally we want to compute V(s) - mean(Q(s,a)) for all a generated
        # here, for simplicity we compute V(s) - best Q(s,a)
        expected_v = task_record.V_next
        actual_Qsa = task_record.Q[1:]
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

    def _get_state_or_sa_to_reflect(self, task_record: TaskRecord, used_idx: set):
        state_scores_ = self._selection_metric(task_record)
        most_unexpected_idx_sorted = np.argsort(state_scores_)[::-1]

        all_actions = []
        all_states = []
        for data in task_record.trajectory:
            if isinstance(data, Action):
                all_actions.append(data)
            else:
                all_states.append(data)

        found_idx = -1
        for i in most_unexpected_idx_sorted:
            if i not in used_idx:
                found_idx = i
                break
        assert found_idx != -1, "No more state to reflect on"

        trajectory = task_record.trajectory
        end_idx = found_idx * 2 + 2 # +2 because we start after the first (s,a)
        partial_trajectory = trajectory[:end_idx+1]
        final_score = task_record.final_score if self.use_gt_success else task_record.est_final_score
        V_estimated = task_record.V_next[found_idx]

        debate_performed = task_record._debates[found_idx]
        metadata = {
            "intent": task_record.task_info["intent"],
            "intent_images": task_record.task_info["images"],
            "V": V_estimated,
            "debate": debate_performed,
        }
        return (
            partial_trajectory,
            final_score,
            metadata
        )

    @time_it
    def reflect(self, task_record: TaskRecord) -> list[DebateReflectionRecord]:
        # ask llm what it expect the agent to do, and what will happen if we let the agent cook
        # compare that with the actual trajectory and task status. gen reflection about agent's ability
        """Reflect on the task and return lessons learned"""
        max_num_records = self.max_reflections_per_task
        tmp_task_record = copy.deepcopy(task_record)
        # remove the last state so that its always (s,a,s,a,...,s,last_a)
        if isinstance(tmp_task_record.trajectory[-1], dict):
            tmp_task_record.trajectory.pop(-1)
        # adjust Q, V from (q, v_next, q, v_next) to (q, v_next, q, v_next, final score) for _selection_metric
        used_idx = set()
        if self.use_gt_success:
            # RLHF typed
            if tmp_task_record.final_score == 0:
                tmp_task_record.Q.append(-1.0)
            else:
                tmp_task_record.Q.append(1.0)
        else:
            # fully self-supervised
            if tmp_task_record.est_final_score == 0:
                tmp_task_record.Q.append(-1.0)
            else:
                tmp_task_record.Q.append(1.0)
        state_scores_ = self._selection_metric(tmp_task_record)
        debate_data = tmp_task_record._debates
        most_unexpected_idx_sorted = np.argsort(state_scores_)[::-1]
        
        reflection_records = []
        for i in range(max_num_records):
            logger.info(f"Reflecting iteration {i}")
            # we are either evaluating a state (and its previous actions)
            # or the last action (since there is no next state)
            found_idx = -1
            for try_idx in most_unexpected_idx_sorted:
                if try_idx not in used_idx:
                    found_idx = try_idx
                    break
            if found_idx == -1:
                logger.info(f"No more state to reflect on, {used_idx=}")
                break
            most_unexpected_score = state_scores_[found_idx]

            if most_unexpected_score < self.reflection_threshold:
                logger.info(f"unexpected score {most_unexpected_score} is below {self.reflection_threshold=}. Stopping reflection.")
                break
            # check if debate is there
            if "opposing_reasons" not in debate_data[found_idx] or "supporting_reasons" not in debate_data[found_idx]:
                logger.info(f"Missing debate data for {found_idx=}. Debate data={debate_data[found_idx]}. Skipping reflection.")
                used_idx.add(found_idx)
                continue
            
            (
                partial_trajectory,
                final_score,
                metadata
            ) = self._get_state_or_sa_to_reflect(tmp_task_record, used_idx)

            reflection = self._get_reflections(
                partial_trajectory,
                final_score,
                metadata
            )
            logger.info(f"reinforced debate value function generated reflection: {reflection}")

            ##### save reflection
            # first, caption the images which we will use as (text-based) retrieval
            # note that these caption calls should be cached (by calling generate_rubrics)
            intent_images_text = []
            intent_images = metadata["intent_images"]
            if intent_images is not None:
                for i_img in intent_images:
                    intent_images_text.append(self._caption_image(i_img))
            debate_performed = metadata["debate"]

            ### clean some metadata to save space
            new_reflection_record = DebateReflectionRecord(
                intent=metadata["intent"],
                intent_images=metadata["intent_images"],
                _intent_images_text=intent_images_text,
                partial_trajectory_to_eval=partial_trajectory,
                gen_supporting_reasons=debate_performed["supporting_reasons"],
                gen_opposing_reasons=debate_performed["opposing_reasons"],
                estimated_V=metadata["V"],
                # other stuff
                reflection=reflection,
                _task_success=final_score == 1.0,
                _from_task_hash=hash(task_record)
            )
            reflection_records.append(new_reflection_record)

            # update to prepare for next iteration
            used_idx.add(found_idx)
        return reflection_records

    @time_it
    def on_task_end(
        self,
        trajectory: Trajectory,
        score: float,
        task_info: dict,
        meta_data: Any,
        search_tree_stats: None,
        debate_data: None,
        **kwargs
    ) -> None:
        assert search_tree_stats is not None, "search_tree_stats field is missing"
        assert debate_data is not None, "debate_data field is missing"
        Q = search_tree_stats["Q"]
        Nsa = search_tree_stats["Nsa"]
        P = search_tree_stats["P"]
        V_next = search_tree_stats["V_next"]

        logger.debug(f'received debate_data with lengths={[len(d) for d in debate_data]}')

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
            _debates=debate_data
        )
        reflection_records = self.reflect(new_task_record)

        # save reflection record
        all_reflections: list[DebateReflectionRecord] = self._load_lzma_db_files(self.reflection_folder_path)
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


class ReinforcedDebateValueFunctionNoREFLECTION(ReinforcedDebateValueFunction):
    # ablation study
    @time_it
    def on_task_end(
        self,
        trajectory: Trajectory,
        score: float,
        task_info: dict,
        meta_data: Any,
        search_tree_stats: None,
        debate_data: None,
        **kwargs
    ) -> None:
        assert search_tree_stats is not None, "search_tree_stats field is missing"
        assert debate_data is not None, "debate_data field is missing"
        Q = search_tree_stats["Q"]
        Nsa = search_tree_stats["Nsa"]
        P = search_tree_stats["P"]
        V_next = search_tree_stats["V_next"]

        logger.debug(f'received debate_data with lengths={[len(d) for d in debate_data]}')

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
            _debates=debate_data
        )
        # reflection_records = self.reflect(new_task_record)

        # save reflection record
        logger.info(f"Skipping reflection for ablation study")

        # save task record in case we need to do analysis later
        all_task_records: list[TaskRecord] = self._load_lzma_db_files(self._task_record_folder_path)
        logger.info(f"Found {len(all_task_records)} task records from {self._task_record_folder_path}")
        existing_task_hashes = set([hash(r) for r in all_task_records])
        new_task_record_to_write = [r for r in [new_task_record] if hash(r) not in existing_task_hashes]
        logger.info(f"Deduped and writing {len(new_task_record_to_write)} new task records to {self._task_record_folder_path}")
        self._write_lzma_db_files(self._task_record_folder_path, new_task_record_to_write)
        return
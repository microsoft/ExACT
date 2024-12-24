## this script is experimental.
## Please refer to tree_to_data.ipynb in case of any issues.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import json
import pickle
import lzma
import numpy as np
import math
import random
import jsonlines
import torch
import argparse
import requests
import copy
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from cachetools import Cache
from PIL import Image
from browser_env.utils import StateInfo, pil_to_b64
from src.helper_functions import get_action_description
from src.constants import SIMPLE_LLM_API_CACHE
from src.evaluation import image_utils
from src.agentic.policy import CoTPolicyPConstructor, ExploratoryCoTPolicyPConstructor
from src.llms.lm_config import LMConfig
from src.llms.tokenizer import Tokenizer
from src.agentic.value_function import create_chat_completion_wrapper
from src.envs.actions import Action, ActionTypes
from src.agent.mcts_agent import Node
from browser_env.env_config import URL_MAPPINGS
from collections import defaultdict


DSET_NAME_TO_FOLDER = {
    "classifields": "configs/visualwebarena/test_classifieds_v2",
    "reddit": "configs/visualwebarena/test_reddit_v2",
    "shopping": "configs/visualwebarena/test_shopping_v2"
}

# since we need to do some rephrasing, we need access to an LLM
assert os.get("VALUE_FUNC_PROVIDER", None) is not None, "VALUE_FUNC_PROVIDER not set"
assert os.get("VALUE_FUNC_API_BASE", None) is not None, "VALUE_FUNC_API_BASE not set"

llm_config = LMConfig(
    provider=os.environ['VALUE_FUNC_PROVIDER'],
    model="gpt-4o",
    mode="chat"
)
llm_config.gen_config["temperature"] = 1.0
llm_config.gen_config["top_p"] = 0.95
llm_config.gen_config["context_length"] = 0
llm_config.gen_config["max_tokens"] = 384
llm_config.gen_config["stop_token"] = None
llm_config.gen_config["max_obs_length"] = 3840
llm_config.gen_config["max_retry"] = 1
llm_tokenizer = Tokenizer(
    model_name="gpt-4o",
    provider=os.environ['VALUE_FUNC_PROVIDER']
)


def configure_captioning_fn():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    caption_image_fn = image_utils.get_captioning_fn(
        device, dtype, "Salesforce/blip2-flan-t5-xl"
    )
    return caption_image_fn


captioning_fn = configure_captioning_fn()


def cached_caption_image_fn(images: list):
    encoded_images_str = ""
    for image in images:
        encoded_images_str += pil_to_b64(image)
    if encoded_images_str in IMAGE_CAPTION_CACHE:
        return IMAGE_CAPTION_CACHE[encoded_images_str]
    
    captions = captioning_fn(images)
    IMAGE_CAPTION_CACHE[encoded_images_str] = captions
    return captions


IMAGE_CAPTION_CACHE = Cache(maxsize=1000)

cache_save_path = "ft_image_cache.pkl"
# save this cache
if os.path.exists(cache_save_path):
    with open(cache_save_path, "rb") as fread:
        IMAGE_CAPTION_CACHE.update(pickle.load(fread))
    print(f"Loaded {len(IMAGE_CAPTION_CACHE)} cache entries")

def save_image_cache():
    with open(cache_save_path, "wb") as fwrite:
        pickle.dump(IMAGE_CAPTION_CACHE, fwrite)
    print(f"Saved {len(IMAGE_CAPTION_CACHE)} cache entries")
    return


def get_action_descs(trajectory, action_set_tag: str):
    action_strs = ["None"]
    prev_state = None
    for data in trajectory:
        if isinstance(data, dict):
            prev_state = data
        else:
            action = data
            # observation_metadata = prev_state['info']['observation_metadata']
            if 'obs_metadata' not in action.metadata:
                observation_metadata = prev_state['info']['observation_metadata']
            else:
                observation_metadata = action.metadata['obs_metadata']
            action_desc = get_action_description(
                action,
                observation_metadata=observation_metadata,
                action_set_tag=action_set_tag,
                prompt_constructor=None
            )
            action_strs.append(action_desc)
    return action_strs


def format_trajectory_to_chat(prompt_constructor, trajectory, last_action: Action, task_info):
    # make sure the last one is state
    assert isinstance(trajectory[-1], dict)

    images = task_info["images"]  # intent images
    intent  = task_info["intent"]
    meta_data = {}

    action_history_descs = get_action_descs(trajectory, "id_accessibility_tree")
    meta_data["action_history"] = action_history_descs

    # Caption the input image, if provided.
    if images is not None and len(images) > 0:
        image_input_caption = ""
        for image_i, image in enumerate(images):
            if image_i == 0:
                image_input_caption += f'Input image {image_i+1}: "{cached_caption_image_fn([image])[0]}"'
            else:
                image_input_caption += f'input image {image_i+1}: "{cached_caption_image_fn([image])[0]}"'
            if len(images) > 1:
                image_input_caption += ", "
        # Update intent to include captions of input images.
        intent = f"{image_input_caption}\nIntent: {intent}"

    prompt = prompt_constructor.construct(
        trajectory, intent, meta_data  # empty images since we use caption for training data
    )
    ### add final output
    agent_resp = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": last_action.raw_prediction,
            }
        ]
    }
    prompt.append(agent_resp)
    return prompt


def flatten_to_trainable_chat(chat: list, train_last_only=False):
    train_sample = []
    _all_weights = []
    for i, message in enumerate(chat):
        role = message["role"]
        content = message["content"]
        if isinstance(content, list):
            str_contents = []
            for c in content:
                assert c["type"] == "text"
                str_contents.append(c["text"])
            str_content = "\n\n".join(str_contents)
        else:
            assert isinstance(content, str)
            str_content = content
        
        if role == "user":
            train_sample.append({
                "role": role,
                "content": str_content
            })
        elif role == "system":
            train_sample.append({
                "role": role,
                "content": str_content
            })
        elif role == "assistant":
            if train_last_only:
                is_last = i == len(chat) - 1
                train_sample.append({
                    "role": role,
                    "content": str_content,
                    "weight": 1 if is_last else 0
                })
                _all_weights.append(1 if is_last else 0)
            else:
                train_sample.append({
                    "role": role,
                    "content": str_content,
                    "weight": 1
                })
                _all_weights.append(1)
        else:
            raise ValueError(f"Unknown role {role}")

    print(f"formatted traj length {len(train_sample)}, weights: {_all_weights}")
    return {
        "messages": train_sample
    }


def display_trainable_chat(train_sample: dict):
    chat = train_sample["messages"]
    for i, message in enumerate(chat):
        role = message["role"]
        content = message["content"]
        print(f"[[[Turn {i} with {role}]]]")
        print(f"{content}")
        print()
    return


def is_same_element(element_a: dict, element_b: dict):
    a_text = element_a['text'].lower()
    a_text = a_text[a_text.find(' ') + 1:]  # remove the [xxx] in front
    b_text = element_b['text'].lower()
    b_text = b_text[b_text.find(' ') + 1:]

    a_words = set(a_text.split())
    b_words = set(b_text.split())

    num_similar_words = len(a_words.intersection(b_words))
    if num_similar_words / len(a_words) >= 0.75:
        print(f"treating element_a={element_a['text']}, element_b={element_b['text']} as the same.")
        return True
    return False


def maybe_update_action_id(action: Action, info: dict = None) -> Action:
    assert info is not None
    env_obs_metadata = info['observation_metadata']
    env_obs_text_nodes_info_ = env_obs_metadata['text'].get('obs_nodes_info', {})
    env_obs_text_nodes_info = {k: v['text'] for k, v in env_obs_text_nodes_info_.items()}
    env_obs_som_nodes_info = env_obs_metadata['image'].get('obs_nodes_semantic_info', {})

    env_obs_nodes_info = {}
    if len(env_obs_som_nodes_info) > 0:
        env_obs_nodes_info = env_obs_som_nodes_info
    elif len(env_obs_text_nodes_info) > 0:
        env_obs_nodes_info = env_obs_text_nodes_info
    else:
        print(f"maybe_update_action: both text and image has no nodes, skipping")
        return action

    # decide which action obs node it is
    if 'obs_metadata' not in action.metadata:
        print(f"obs_metadata not found in action={action.to_simple_str()}, skippping")
        return action
    
    action_obs_nodes_info = {}
    action_obs_text_nodes_info = action.metadata['obs_metadata'].get('text', {}).get('obs_nodes_info', {})
    action_obs_som_nodes_info = action.metadata['obs_metadata'].get('image', {}).get('obs_nodes_semantic_info', {})
    if len(action_obs_som_nodes_info) > 0:
        action_obs_nodes_info = action_obs_som_nodes_info
    elif len(action_obs_text_nodes_info) > 0:
        action_obs_nodes_info = {k: v['text'] for k, v in action_obs_text_nodes_info.items()}

    
    action_element_id = action.element_id
    if action_element_id == '':
        return action
    if action_element_id not in action_obs_nodes_info:
        print(f"action_element_id={action_element_id} not found in its own nodes={action_obs_nodes_info.keys()}, skipping")
        return action
    
    if action_element_id in env_obs_nodes_info:
        # check if element is matched
        env_node = {
            'text': env_obs_nodes_info[action_element_id]
        }
        action_node = {
            'text': action_obs_nodes_info[action_element_id]
        }
        if is_same_element(env_node, action_node):
            return action
        else:
            print(f"found element might have changed from {action_obs_nodes_info[action_element_id]} to {env_obs_nodes_info[action_element_id]}.")

    print(f'maybe_update_action trying to update action={action.to_simple_str()}')
    print(f'maybe_update_action env_obs_nodes_info={env_obs_nodes_info.keys()}')
    print(f'maybe_update_action action_obs_nodes_info={action_obs_nodes_info.keys()}')
    
    error_margin = int(0.1 * len(action_obs_nodes_info))
    error_margin = max(1, error_margin)
    # assume root node is the min
    action_min_node_id = min([int(k) for k in action_obs_nodes_info.keys()])
    action_element_id_offset = int(action_element_id) - action_min_node_id
    env_min_node_id = min([int(k) for k in env_obs_nodes_info.keys()])

    ## start from middle and search for left and right
    is_updated = False
    for i in range(error_margin+1):
        possible_id = str(action_element_id_offset + env_min_node_id + i)
        if possible_id in env_obs_nodes_info:
            env_node = {
                'text': env_obs_nodes_info[possible_id]
            }
            action_node = {
                'text': action_obs_nodes_info[action_element_id]
            }
            if is_same_element(env_node, action_node):
                # do the substitution
                previous_raw_prediction = action.raw_prediction
                action.metadata['previous_raw_prediction'] = previous_raw_prediction
                action.metadata['previous_element_id'] = action_element_id

                action.element_id = possible_id
                action.metadata['obs_metadata'] = env_obs_metadata
                action.raw_prediction = previous_raw_prediction.replace(f"[{action_element_id}]", f"[{possible_id}]")
                print(f"maybe_update_action updated action={action.to_simple_str()}")
                is_updated = True
                break
        
        possible_id = str(action_element_id_offset + env_min_node_id - i)
        if possible_id in env_obs_nodes_info:
            env_node = {
                'text': env_obs_nodes_info[possible_id]
            }
            action_node = {
                'text': action_obs_nodes_info[action_element_id]
            }
            if is_same_element(env_node, action_node):
                # do the substitution
                previous_raw_prediction = action.raw_prediction
                action.metadata['previous_raw_prediction'] = previous_raw_prediction
                action.metadata['previous_element_id'] = action_element_id

                action.element_id = possible_id
                action.metadata['obs_metadata'] = env_obs_metadata
                action.raw_prediction = previous_raw_prediction.replace(f"[{action_element_id}]", f"[{possible_id}]")
                print(f"maybe_update_action updated action={action.to_simple_str()}")
                is_updated = True
                break
    if not is_updated:
        print(f"maybe_update_action failed to update action.")
    return action


SIMPLE_LLM_API_CACHE = Cache(maxsize=1000)
llm_cache_save_path = "llm_api_cache.pkl"
# save this cache
if os.path.exists(llm_cache_save_path):
    with open(llm_cache_save_path, "rb") as fread:
        SIMPLE_LLM_API_CACHE.update(pickle.load(fread))
    print(f"Loaded {len(SIMPLE_LLM_API_CACHE)} cache entries")


def save_llm_cache():
    with open(llm_cache_save_path, "wb") as fwrite:
        pickle.dump(SIMPLE_LLM_API_CACHE, fwrite)
    print(f"Saved {len(SIMPLE_LLM_API_CACHE)} cache entries")
    return



def map_url_to_real(url: str) -> str:
    """Map the urls to their real world counterparts"""
    for i, j in URL_MAPPINGS.items():
        if i in url:
            url = url.replace(i, j)
    return url


def _filter_train_data(formatted_chat):
    error_kwd = "no matching element found"
    reflection_kwd = "reflections"
    if error_kwd in formatted_chat[-1]["content"]:
        return True # remove
    
    # last assistant turn is not empty (e.g. no errors)
    last_turn = formatted_chat[-1]
    assert last_turn["role"] == "assistant"
    if last_turn["content"].strip() == "":
        return True
    return False


REPHRASE_REFL_PRMOPT = """
Below are some texts that are generated using model self-reflection, which provides hints on how to perform better on a web task.
Please rephrase the following text to make it:
1. sound natural even WITHOUT the word "reflections" appearing in text.
2. to make it sound natural, you can consider converting these reflections/insights into your own thinking (see example 1 below).
3. do NOT alter the overall meaning of the text, as well as the actions inside ```click [xx]```, ```type "xxx"```, etc.
4. do NOT generate anything after rephrasing 2.

For example:
## Original 1:
Given the reflections, it would be more efficient to navigate directly through the relevant category link rather than using the search box. Since we need to find the most recent painting in the "Arts + crafts" category, I will first click on the "Arts + crafts" category link.

In summary, the next action I will perform is ```click [34]```
## Rephrasing 1 without reflections:
Maybe it is more efficient to navigate directly through the relevant category link rather than using the search box. Since we need to find the most recent painting in the "Arts + crafts" category, I will first click on the "Arts + crafts" category link.

In summary, the next action I will perform is ```click [34]```

## Original 2:
{original_text}
## Rephrasing 2 without reflections:
""".strip()


def _rephrase_reflection_content(prediction_str: str):
    try:
        original_exec = re.search(r"```(.+)```", prediction_str).group(1)
        completion = create_chat_completion_wrapper(
            messages = [{
                "role": "user",
                "content": REPHRASE_REFL_PRMOPT.format(original_text=prediction_str.strip())
            }],
            model="gpt-4o",
            temperature=0.7,
            max_tokens=256,
            top_p=0.95,
            num_outputs=1,
        )
        rephrased_exec = re.search(r"```(.+)```", completion).group(1)
        assert original_exec == rephrased_exec, f"Original: {original_exec}, Rephrased: {rephrased_exec}"
        completion = completion.replace("## Rephrasing 2 without reflections:", "").strip()

        print(f"rephrased from {prediction_str}\nto\n{completion}")
    except Exception as e:
        print(f"Error: {e}")
        completion = prediction_str
    return completion


def _find_all_links(text: str) -> list:
    links = []
    # assume http:// or https://, ends with either space or ]
    link_pattern = re.compile(r"https?://[^\s\]]+")
    for match in link_pattern.finditer(text):
        link = match.group()
        links.append(link)
    return links

def _replace_links(text: str) -> str:
    found_links = _find_all_links(text)
    for link in found_links:
        text = text.replace(link, map_url_to_real(link))
    return text


def _trainable_chat_postprocessing(trainable_chat, no_rephrase=False):
    # e.g., rephrase the reflection data
    messages = trainable_chat["messages"]
    for m in messages:
        if m["role"] == "assistant":
            ## rephrase reflection content
            if "reflection" in m["content"].lower() and not no_rephrase:
                m["content"] = _rephrase_reflection_content(m["content"])
            # though raw action data contains real urls, PARAPHRASED REFLECTIONS can sometimes leak it
            m["content"] = _replace_links(m["content"])

    # check last turn
    last_turn = messages[-1]
    assert last_turn["role"] == "assistant"
    assert "reflections" not in last_turn["content"].lower()
    return trainable_chat


def get_single_training_data_from_trajectory(trajectory, tid: int, dset_name: str, prompt_constructor, modality='som_no_image'):
    assert modality in ['som_no_image', 'text']
    traj_before_last_action = trajectory[:-1]
    last_action = trajectory[-1]

    eval_config_file = f"{DSET_NAME_TO_FOLDER[dset_name]}/{tid}.json"
    if not os.path.exists(eval_config_file):
        raise Exception(f"Cannot find {eval_config_file}")
    
    with open(eval_config_file, "r") as fread:
        eval_config = json.load(fread)

    images = []
    intent_image = eval_config.get("image", None)
    if intent_image is not None:
        if not isinstance(intent_image, list):
            image_paths = [intent_image]
        else:
            image_paths = intent_image
        
        for image_path in image_paths:
            # Load image either from the web or from a local path.
            if image_path.startswith("http"):
                req = requests.get(
                    image_path,
                    headers={"User-Agent": "Mozilla/5.0"},
                    stream=True
                )
                input_image = Image.open(req.raw)
            else:
                input_image = Image.open(image_path)
            images.append(input_image)
    task_info = {
        "intent": eval_config["intent"],
        "images": images
    }

    print('Using prompt constructor:', prompt_constructor)

    if modality == 'text':
        chat_histroy = format_trajectory_to_chat(
            prompt_constructor,
            traj_before_last_action,
            last_action,
            task_info
        )
    else:
        raise NotImplementedError(f"Unknown modality {modality}")

    trainable_chat = flatten_to_trainable_chat(chat_histroy, train_last_only=True)
    return trainable_chat


def check_format_errors(dataset: list[dict]):
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")
    return


def find_successful_traj_w_trees(base_dir: str):
    search_tree_dir = os.path.join(base_dir, "search_trees")
    perf_dir = os.path.join(base_dir, "performances")
    log_file_dir = os.path.join(base_dir, "log_files")  # check if task is correct
    traj_file_dir = os.path.join(base_dir, "trajectories")

    found_data = []
    for found_search_trees in os.listdir(search_tree_dir):
        # check if it is a folder
        search_tree_folder = os.path.join(search_tree_dir, found_search_trees)
        if not os.path.isdir(search_tree_folder):
            continue
        # e.g., task_101/
        task_id = int(found_search_trees.split("_")[-1])

        ### 1. check if it is successful
        perf_file = os.path.join(perf_dir, f"performance_{task_id}.json")
        log_file = os.path.join(log_file_dir, f"task_{task_id}.log.txt")
        if os.path.exists(perf_file):
            # if has perf file, use it
            with open(perf_file, "r") as fread:
                perf = json.load(fread)

            success = perf["scores"] == 1.0
            if not success:
                continue
        elif os.path.exists(log_file):
            # if has log file, use it
            with open(log_file, "r") as fread:
                log = fread.read()

            success = "[Result] (PASS)" in log
            if not success:
                continue
        else:
            print(f"Log or perf file not found for {task_id}")
            continue
        
        ### obtain the actual executed trajectory file
        traj_fpath = os.path.join(traj_file_dir, f"task_{task_id}.pkl.xz")
        if not os.path.exists(traj_fpath):
            print(f"Trajectory file not found for {task_id}")
            continue
        with lzma.open(traj_fpath, "rb") as fread:
            traj = pickle.load(fread)

        all_trees_pickle_files = []
        for file in os.listdir(search_tree_folder):
            if file.endswith(".pkl.xz"):
                all_trees_pickle_files.append(file)
        # sort by date
        # tree_20240921-225009.pkl.xz
        # convert 20240921-225009 to timestamp
        all_time_stamps = []
        for tree_file in all_trees_pickle_files:
            time_stamp = tree_file.split("_")[-1].split(".")[0]
            all_time_stamps.append(time_stamp)
        sorted_idx = np.argsort(all_time_stamps)
        sorted_trees = []
        for i in sorted_idx:
            tree_fpath = os.path.join(search_tree_folder, all_trees_pickle_files[i])
            with lzma.open(tree_fpath, "rb") as fread:
                tree = pickle.load(fread)
            sorted_trees.append(tree)
        

        found_data.append({
            "task_id": task_id,
            "traj": traj,
            "trees": sorted_trees
        })
    print(f"Found {len(found_data)} successful tasks")
    return found_data


def _fake_simulation(state: Node):
    # no-op since we have already done the simulation
    return

def _fake_expansion(state: Node, Ns, Nsa, Q):
    # add back children
    state.children = state._children

    hashable_state = state._to_string_rep()

    Ns[hashable_state] = 0
    Nsa[hashable_state] = defaultdict(lambda: 0.0)
    Q[hashable_state] = defaultdict(lambda: 0.0)  # 0.0 for Q[s][new_a]
    # P is already precomputed
    return


def _replay_tree_traversal(
    state: Node,
    traversal_buffer: list,
    Q: dict,
    Ns: dict,
    Nsa: dict
):
    # replay MCTS tree traversal until end_action is hit
    hashable_state = state._to_string_rep()
    
    v = 0.0
    # if this leaf node is terminal, return the value
    if state.is_terminal:
        # terminal node
        if state._need_evaluation:
            _fake_simulation(state)
        return state.value
    elif state.value == 1.0:
        return state.value
    elif len(state.children) == 0:
        # selected leaf node, expand and simulate (for backprop below)
        _fake_expansion(state, Ns, Nsa, Q)
        _fake_simulation(state)
        return state.value
    
    ##### Selection
    # existing, continue selection
    # go next state by picking best according to U(s,a)
    cpuct = 1.0
    best_uct = -float('inf')
    best_action = None
    for a in state.children.keys():
        _Ns = Ns[hashable_state]
        _qsa = Q[hashable_state][a]
        _p = a.metadata["P"]
        _nsa = Nsa[hashable_state][a]
        if Ns == 0:  # first time visit
            uct = _qsa + cpuct * _p
        else:
            uct = _qsa + cpuct * _p * math.sqrt(_Ns) / (1 + _nsa)
        
        if uct > best_uct:
            best_uct = uct
            best_action = a
            print(f"updating best action: {best_action.raw_prediction}")
            print(f"uct={uct} (with {_Ns=}, {_nsa=}, {_qsa=}, {_p=})")
    print(f"selected best action: {best_action.raw_prediction}")
    
    # transition and update that state's metadata
    # best_action_cache.append(best_action)
    # next_state = await self._get_next_state(state, best_action)
    next_state = state.children[best_action]
    
    ##### Expansion and Simulation
    # 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
    # 2. if leaf, we will expand it and return the value for backpropagation
    v = _replay_tree_traversal(
        next_state,
        traversal_buffer,
        Q,
        Ns,
        Nsa
    )

    ##### Backpropagation
    # update stats
    # add in new estimate and average
    Q[hashable_state][best_action] = (Nsa[hashable_state][best_action] * Q[hashable_state][best_action] + v) / (Nsa[hashable_state][best_action] + 1)
    print(f"backpropagating value {v} to get Q[{hashable_state}][{best_action.raw_prediction}]={Q[hashable_state][best_action]}")
    Nsa[hashable_state][best_action] += 1
    Ns[hashable_state] += 1
    state.Ns += 1

    # update metadata in action
    best_action.metadata["Q"] = Q[hashable_state][best_action]
    best_action.metadata["Nsa"] = Nsa[hashable_state][best_action]
    best_action.metadata["V_next"] = next_state.value
    
    if len(next_state.trajectory) == 0:
        traversal_buffer.insert(1, next_state)  # the 0ths root state should be untouched
    else:
        traversal_buffer.insert(1, next_state.trajectory[-1])  # the 0ths root state should be untouched
    traversal_buffer.insert(1, best_action)

    ### check if action is found
    element_id = best_action.element_id
    if element_id != '':
        if not state.is_terminal:
            elemnt_text = f"[{element_id}]"
            state_dict = state.trajectory[-1]
            state_obs_text = state_dict['observation']['text']
            action_nodes = best_action.metadata['obs_metadata']['text']['obs_nodes_info']
            if elemnt_text in state_obs_text:
                print(f"{elemnt_text=} is found in current state")
            else:
                print(f"{elemnt_text=} is NOT found in current state")
                if element_id in action_nodes:
                    print(f"{element_id=} is found in action nodes")
                else:
                    print(f"{element_id=} is NOT found in action nodes")
    return v


def _prepare_fake_tree(root: Node):
    # temporary reset all stats
    root._Ns = root.Ns
    root.Ns = 0

    all_a_s = root._get_all_child_actions()
    for a, s in all_a_s:
        s._Ns = s.Ns
        s.Ns = 0
        s._children = s.children
        s.children = {}
    
    root._children = root.children
    root.children = {}
    return root

def __non_numeric_words(state_text):
    words = []
    for w in state_text.split():
        if re.match(r"\[\d+\]", w):
            continue
        words.append(w)
    return words


def _is_state_similar(state1, state2):
    if state1 is None and state2 is None:
        print('received two None actions')
        return True
    if state1 is None or state2 is None:
        return False
    state1_text = state1['observation']['text']
    state2_text = state2['observation']['text']
    state_1words = __non_numeric_words(state1_text)
    state_2words = __non_numeric_words(state2_text)
    
    num_overlap_words = len(set(state_1words).intersection(set(state_2words)))
    total_words = len(set(state_1words).union(set(state_2words)))
    print(f"overlap words: {num_overlap_words}, total words: {total_words}, ratio: {num_overlap_words / total_words}")
    return num_overlap_words / total_words > 0.9


def _is_action_similar(action1: Action, action2: Action):
    if action1 is None and action2 is None:
        print('received two None actions')
        return True
    if action1 is None or action2 is None:
        return False
    action1_text = action1.raw_prediction
    action1_element_id = action1.element_id
    action2_text = action2.raw_prediction
    action2_element_id = action2.element_id
    action1_text = action1_text.replace(f"[{action1_element_id}]", "[X]")
    action2_text = action2_text.replace(f"[{action2_element_id}]", "[X]")

    print(f'checking similarity between\n{action1_text}\nand\n{action2_text}')
    print('result, they are similar:', action1_text.strip() == action2_text.strip())
    return action1_text.strip() == action2_text.strip()


def _is_trajectory_similar(traj1, traj2):
    if len(traj1) != len(traj2):
        return False
    
    for data1, data2 in zip(traj1, traj2):
        if isinstance(data1, dict) and isinstance(data2, dict):
            if not _is_state_similar(data1, data2):
                return False
        elif isinstance(data1, Action) and isinstance(data2, Action):
            if not _is_action_similar(data1, data2):
                return False
        elif isinstance(data1, Node) and isinstance(data2, Node):
            if data1.is_terminal != data2.is_terminal:
                return False
            if data1.is_terminal == data2.is_terminal:
                print('both are terminal, assuming same state')
                return True
            if not _is_state_similar(data1.trajectory[-1], data2.trajectory[-1]):
                return False
        else:
            return False
    return True


def replay_tree_traversal(single_tree: Node, traversal_buffer: list, end_action: Action):
    Q = {}  # new stats
    Ns = {}  # new stats
    Nsa = {}  # new stats

    found_last_action = False

    tmp_tree = copy.deepcopy(single_tree)

    tmp_tree = _prepare_fake_tree(tmp_tree)

    itr = 50
    start_state = tmp_tree.trajectory[-1]
    _replay_tree_traversal(tmp_tree, [], Q, Ns, Nsa)  # init root node
    print('traversal start, looking for ', end_action.raw_prediction)
    print(f'root node has {len(single_tree.children)} children')
    while not found_last_action:
        curr_traversal = [start_state]
        _replay_tree_traversal(tmp_tree, curr_traversal, Q, Ns, Nsa)
        assert len(curr_traversal) >= 2, f"curr_traversal: {curr_traversal}"
        if len(curr_traversal) >= 2:
            last_action = curr_traversal[-2]
            if _is_action_similar(last_action, end_action):
                found_last_action = True
                print(f'found last action in {50-itr+1} iterations')
        # curr_traversal.pop()  # we want to end with (s,a,s)

        traversal_buffer.append(curr_traversal)
        itr -= 1
        if itr < 0:
            print("Max iteration reached")
            break
    return traversal_buffer


def __print_traj(traj):
    concat_str = 'None'
    for data in traj:
        if isinstance(data, Action):
            concat_str += f"\n---->\n{data.raw_prediction}"
    print(concat_str)
    return


def __find_common_ancestor_idx(trav1: list, trav2: list):
    found_idx = 0
    for idx, (data_1, data_2) in enumerate(zip(trav1, trav2)):
        try:
            assert type(data_1) == type(data_2)
        except AssertionError:
            raise Exception(f"Data type mismatch: {type(data_1)} != {type(data_2)}")

        if isinstance(data_1, dict):
            if _is_state_similar(data_1, data_2):
                ancestor_state = data_1
                found_idx = idx
        elif isinstance(data_1, Action):
            continue
        elif isinstance(data_1, Node):
            print('type(data_1):', data_1.is_root)
            __print_traj(data_1.trajectory)
            print('type(data_2) is root?', data_2.is_root)
            __print_traj(data_2.trajectory)
            if data_1.is_terminal and data_2.is_terminal:
                # the only allowed possibility is that we are exploring two end actions
                prev_action_1: Action = trav1[idx-1]
                prev_action_2: Action = trav2[idx-1]
                if prev_action_1.action_type != ActionTypes.STOP:
                    raise Exception("Both are terminal, should not happen")
                if prev_action_2.action_type != ActionTypes.STOP:
                    raise Exception("Both are terminal, should not happen")
                found_idx = idx - 1  # last state before stop
                break
            if data_1.is_terminal or data_2.is_terminal:
                found_idx = idx - 1  # last state before stop
                break
            state_1 = data_1.trajectory[-1]
            state_2 = data_2.trajectory[-1]
            if _is_state_similar(state_1, state_2):
                ancestor_state = state_1
                found_idx = idx
    return found_idx

def _find_common_ancestor(trav1: list, trav2: list):
    ancestor_state_idx = __find_common_ancestor_idx(trav1, trav2)
    ancestor_state = trav1[ancestor_state_idx]
    return ancestor_state


def _fastforward_target_to_common_ancestor(trav1: list, trav2: list):
    # same loop as find ancestor, but return the remaining of trav2
    trav2_no_laststate = trav2[:-1]
    skipped_idx = __find_common_ancestor_idx(trav1, trav2_no_laststate)
    return trav2[skipped_idx+1:]  # start with action


def _backtrack_to_common_ancestor(traversal: list, trav2: list, flattened_traj_so_far: list):
    # same loop as find ancestor, but return the remaining of trav2
    trav2_no_laststate = trav2[:-1]
    skipped_idx = __find_common_ancestor_idx(traversal, trav2_no_laststate)
    # (s0, a0, s1, a1, s2), ancestor s1, then this is (s1, a1)
    action_state_to_reverse = traversal[skipped_idx:-1]  # keep first state and remove last state
    ancestor_state = traversal[skipped_idx]

    num_actions_to_reverse = 0
    num_stop_actions_to_reverse = 0
    for i, data in enumerate(action_state_to_reverse):
        if isinstance(data, Action):
            direction = getattr(data, '_direction', '')
            if direction == 'backtrack':
                if getattr(data, '_n_action_reversed', None) is None:
                    raise Exception("_n_action_reversed not inside action")
                num_actions_to_reverse -= data._n_action_reversed
            else:
                num_actions_to_reverse += 1
            # special case
            action_type = data.action_type
            if action_type == ActionTypes.STOP:
                num_stop_actions_to_reverse += 1
    if num_actions_to_reverse < 0:
        raise Exception("Negative number of actions to reverse")
    
    if num_actions_to_reverse == 0:
        return flattened_traj_so_far
    elif num_actions_to_reverse > 1:
        # jump to url
        tmp_action: Action = copy.deepcopy(action_state_to_reverse[1])
        state_to_go_to = copy.deepcopy(ancestor_state)
        if not isinstance(state_to_go_to, dict):
            # cannt be right
           raise Exception("state_to_go_to is not a state")

        # remove stop actions
        if num_stop_actions_to_reverse == 1:
            # only makes sense if its the last action
            last_action_dtype = action_state_to_reverse[-1].action_type
            if last_action_dtype != ActionTypes.STOP:
                raise Exception("STOP action is not the last action")
            flattened_traj_so_far.pop()  # pop node and action in traversal and directly goto
            flattened_traj_so_far.pop()
        elif num_stop_actions_to_reverse > 1:
            raise Exception("More than one stop action to reverse")
        
        raw_url = state_to_go_to['url']
        real_url = map_url_to_real(raw_url)
        state_to_go_to['_direction'] = 'backtrack'
        tmp_action.raw_prediction = f"```goto [{real_url}]```"
        tmp_action.action_type = ActionTypes.GOTO_URL
        tmp_action.element_id = ''
        tmp_action._n_action_reversed = num_actions_to_reverse
        tmp_action._direction = 'backtrack'
        tmp_action._backtrack_url = real_url
        
        flattened_traj_so_far.append(tmp_action)
        flattened_traj_so_far.append(state_to_go_to)
    else:
        # one action, just do ```go_back```
        # action_state_afterwards contains (a->s)
        # may need to deal with STOP action -> end state
        action: Action = copy.deepcopy(action_state_to_reverse[1])
        state = action_state_to_reverse[0]

        if action.action_type == ActionTypes.NONE:
            # beautiful, no-op
            return flattened_traj_so_far
        elif action.action_type == ActionTypes.STOP:
            # remove action from traversal
            if len(traversal) < 2:
                raise Exception(f"WTF is inside traversal??? {traversal=}")
            if traversal[-2].action_type != ActionTypes.STOP:
                raise Exception("STOP action is not followed by STOP action")
            flattened_traj_so_far.pop() # pop state
            flattened_traj_so_far.pop() # pop action
        else:
            # normal action
            action._direction = 'backtrack'
            action._n_action_reversed = 1
            action.element_id = ''
            flattened_traj_so_far.append(action)
            state['_direction'] = 'backtrack'
            flattened_traj_so_far.append(state)
    return flattened_traj_so_far


def post_process_traversal(traversal: list):
    if len(traversal) == 1:
        # next action is the best
        flattened = traversal[0]
        assert len(flattened) == 3
        assert not isinstance(flattened[-1], Action)
        flattened.pop()  # remove the last state
        assert isinstance(flattened[-1], Action)
        return flattened
    
    ### remove consecutive, duplicate trajectories. These correspond to many iterations to reduce exploitation
    new_traversal = [traversal[0]]
    prev_trav = traversal[0]
    for i in range(1, len(traversal)):
        curr_trav = traversal[i]
        # if prev_trav != curr_trav:
        if not _is_trajectory_similar(prev_trav, curr_trav):
            new_traversal.append(curr_trav)
        prev_trav = curr_trav

    ### work with this traversal, so that its a single list of (s,a,s,a,...,a)
    flattened_w_backtrack = new_traversal[0]  # start with (s,a,s)
    
    ## algo:
    ## no backtracking when current action's parent == previous action
    ## backtracking occurs otherwise -> find current action and prev action's common ancestor
    ##   replay prev_action back to that ancestor,
    ##   play current action from that ancestor
    for i in range(1, len(new_traversal)):
        curr_trav = new_traversal[i]

        print(f'working on prev traj {len(flattened_w_backtrack)=}')
        __print_traj(flattened_w_backtrack)
        print(f'working on curr traj {len(curr_trav)=}')
        __print_traj(curr_trav)

        # find common ancestor state
        prev_trav = copy.deepcopy(new_traversal[i-1])
        common_ancestor_state = _find_common_ancestor(prev_trav, curr_trav[:-1])
        
        _backtrack_to_common_ancestor(prev_trav, curr_trav, flattened_traj_so_far=flattened_w_backtrack)
        curr_trav = _fastforward_target_to_common_ancestor(prev_trav, curr_trav)
        
        for data in curr_trav:
            if isinstance(data, tuple):
                data, _ = data
            ### check if we found it already
            if isinstance(data, dict):
                data['_direction'] = 'forward'
            else:
                data._direction = 'forward'
            flattened_w_backtrack.append(data)
    return flattened_w_backtrack


def process_single_traj(traj, trees: list):
    # check num states we can replay
    s_a_pairs_to_replay = []
    for i in range(0, len(traj), 2):
        if i+1 >= len(traj):
            break
        s_a_pairs_to_replay.append((traj[i], traj[i+1]))
    if len(s_a_pairs_to_replay) != len(trees):
        # maybe there is fastforwarding due to V=1.0
        last_action = s_a_pairs_to_replay[-1][1]
        V_next = last_action.metadata.get("V_next", 0.0)
        assert V_next == 1.0, f"V_next={V_next}"

        s_a_pairs_to_replay_ = []
        # fix the traversal
        i = -1
        for i in range(len(trees)-1):
            s_a_pairs_to_replay_.append(s_a_pairs_to_replay[i])
        fast_state = s_a_pairs_to_replay[i+1][0]
        s_a_pairs_to_replay_.append((fast_state, last_action))

        s_a_pairs_to_replay = s_a_pairs_to_replay_
    
    # replay the tree traversal
    all_traversals = []
    for s_a, tree in zip(s_a_pairs_to_replay, trees):
        state, action = s_a
        traversal_buffer = []
        # check if state is the same as tree's root
        state_text = state['observation']['text']
        tree_text = tree.trajectory[-1]['observation']['text']
        assert _is_state_similar(state, tree.trajectory[-1]), f"{state_text}\nNEQ\n{tree_text}"
        
        replay_tree_traversal(tree, traversal_buffer, action)
        traversal_buffer = post_process_traversal(traversal_buffer)
        all_traversals.append(traversal_buffer)
    return all_traversals


BAKCTRACK_BACK = """
Let's see what we have got. {judge_reasoning}

We should take a step back and explore some other options.

In summary, the next action I will perform is ```go_back```.
""".strip()


BAKCTRACK_SCROLL_UP = """
Let's see what we have got. {judge_reasoning}

We should take a step back and explore some other options.

In summary, the next action I will perform is ```scroll [up]```.
""".strip()


BAKCTRACK_SCROLL_DOWN = """
Let's see what we have got. {judge_reasoning}

We should take a step back and explore some other options.

In summary, the next action I will perform is ```scroll [down]```.
""".strip()


BAKCTRACK_GOTO_URL = """
Let's see what we have got. {judge_reasoning}

We should take a step back and explore some other options.

In summary, the next action I will perform is ```goto [{real_url}]```.
""".strip()


FORWARD_PROMPT = """
Let's see what we have got. {judge_reasoning}

{raw_action_prediction}
""".strip()


REPHRASE_BACKTRACK_PRMOPT_W_STATE = """
Below are some texts that are generated using model-model interaction in the format of:
```
OBJECTIVE: <user intent>
OBSERVATION: <observation>
ACTION:
Let's see what we have got. <Success estimate of current progress>

<action reasoning>
```
where the success estimate is generated by a separate model.

Your task is to:
1. rephrase the text to make it look like its generated by a single model, WITHOUT using the word "the agent".
2. The <success estimate> should be COHERENT with the current <action reasoning>. If there are conflicts, you can MODIFY the <success estimate> to make it coherent.
3. If the <success estimate> seems ambiguous given the current <observation>, assume that its correct and simply PARAPHRASE it to make it COHERENT.
4. make sure you keep the keywords such as "Let's see what we have got" and "In summary" intact.
5. do NOT generate aything after rephrasing 2.

For example:
OBJECTIVE 1: Find the mileage of the red car in the second row.
OBSERVATION 1: (omitted for brevity)
## ACTION 1:
Let's see what we have got. The agent's actions did not fulfill the user's intent of finding the mileage of the red car in the second row, which is the 1987 Porsche 911 Carrera. Instead, the agent navigated to the page of a different car, the 2010 Lincoln MKT, making the current state irrelevant to the user's request.

We should take a step back and explore some other options.

In summary, the next action I will perform is ```go_back```.
## Rephrasing ACTION 1 as a standalone, coherent thought:
Let's see what we have got. The current observation doesn't seem useful at fulfilling the user's intent of finding the mileage of the red car in the second row, which is the 1987 Porsche 911 Carrera. Instead, the agent navigated to the page of a different car, the 2010 Lincoln MKT, making the current state irrelevant to the user's request.

We should take a step back and explore some other options.

In summary, the next action I will perform is ```go_back```.

OBJECTIVE 2: {objective}
OBSERVATION 2: {observation}
## ACTION 2:
{original_text}
## Rephrasing ACTION 2 as a standalone, coherent thought:
""".strip()


REPHRASE_FORWARD_PRMOPT_W_STATE = """
Below are some texts that are generated using model-model interaction in the format of:
```
OBJECTIVE: <user intent>
OBSERVATION: <observation>
Let's see what we have got. <Success estimate of current progress>

<action reasoning>
```
where the success estimate is generated by a separate model.

Your task is to:
1. rephrase the text to make it look like its generated by a single model, WITHOUT using the word "the agent".
2. The <success estimate> should be COHERENT with the current <action reasoning>. If there are conflicts, you can MODIFY the <success estimate> to make it coherent.
3. If the <success estimate> seems ambiguous given the current <observation>, assume that its correct and simply PARAPHRASE it to make it COHERENT.
4. make sure you keep the keywords such as "Let's see what we have got" and "In summary" intact.
5. do NOT generate aything after rephrasing 2.

For example:
OBJECTIVE 1: Find me a pillow with an animal pattern.
OBSERVATION 1: (omitted for brevity)
## Original 1:
Let's see what we have got. I have navigated to the listings page for West Virginia, which is necessary to find the most expensive green vehicle. However, I have not yet filtered the listings by "Cars + trucks" and identified the green vehicle with the highest price, nor provided the lister's name. Therefore, the task is not complete and needs further actions.

Let's think step-by-step.

1. The listings have been filtered to those in West Virginia.
2. Searching for green vehicles is the next step. The current listings show various items, including some vehicles.
3. The task is to find the most expensive green vehicle. Among the listings, the "1988-1998 Chevy 1500, 2500 6ft" appears to be a green vehicle priced at $1200.

Next, I will check the listing for this vehicle to confirm its color and find the lister's name. In summary, the next action I will perform is ```click [1028]```
## Rephrasing 1 as a standalone, coherent thought:
Let's see what we have got. I have navigated to the listings page for West Virginia, which is necessary to find the most expensive green vehicle. However, I have not yet identified the green vehicle with the highest price on this page. Therefore, the task is not complete and needs further actions.

I think I should proceed with the following steps:

1. The listings have been filtered to those in West Virginia.
2. Searching for green vehicles is the next step. The current listings show various items, including some vehicles.
3. The task is to find the most expensive green vehicle. Among the listings, the "1988-1998 Chevy 1500, 2500 6ft" appears to be a green vehicle priced at $1200.

Next, I will check the listing for this vehicle to confirm its color and find the lister's name. In summary, the next action I will perform is ```click [1028]```

OBJECTIVE 2: {objective}
OBSERVATION 2: {observation}
## ACTION 2:
{original_text}

Note that making the <success estimate> COHERENT with the current <action reasoning> is the MOST important. If needed, you can MODIFY the <success estimate> to make it coherent.
## Rephrasing ACTION 2 as a standalone, coherent thought:
""".strip()


def _rephrase_success_estimate_v2(intent, state, prediction_str: str, mode='backtrack'):
    try:
        original_exec = re.search(r"```(.+)```", prediction_str).group(1)
        
        if mode == 'backtrack':
            content = REPHRASE_BACKTRACK_PRMOPT_W_STATE.format(
                objective=intent,
                observation=state,
                original_text=prediction_str.strip()
            )
        elif mode == 'forward':
            content = REPHRASE_FORWARD_PRMOPT_W_STATE.format(
                objective=intent,
                observation=state,
                original_text=prediction_str.strip()
            )
        else:
            raise Exception(f"Unknown mode: {mode}")
        
        completion = create_chat_completion_wrapper(
            messages = [{
                "role": "user",
                "content": content
            }],
            model="gpt-4o",
            temperature=0.7,
            max_tokens=256,
            top_p=0.95,
            num_outputs=1,
        )
        rephrased_exec = re.search(r"```(.+)```", completion).group(1)
        assert original_exec == rephrased_exec, f"Original: {original_exec}, Rephrased: {rephrased_exec}"

        completion = completion.replace("## Rephrasing 2 as a standalone thought:", "").strip()
        print(f"rephrased from {prediction_str}\nto\n{completion}")
    except Exception as e:
        print(f"Error: {e}")
        completion = prediction_str
    return completion


IDX_TO_ALPHABET = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
}
ALPHABET_TO_IDX = {v: k for k, v in IDX_TO_ALPHABET.items()}


def __format_mcq_choices(choices):
    formatted_choices = []
    for idx, choice in enumerate(choices):
        formatted_choices.append(f"{IDX_TO_ALPHABET[idx]}: {choice}")
    return '\n'.join(formatted_choices)


CHOOSE_JUDGE_PROMPT = """
Below is a user intent, a text representation of a webpage, and an agent's action execution. The goal of the agent is to fulfill the user intent by executing the correct action.
To evaluate the current progress and future success (i.e., potential) of curernt action, we have several judgements from different agents.
````
OBJECTIVE: <user intent>
OBSERVATION: <text representation of a webpage>
ACTION: <agent's action execution>

JUDGES:
<list of judgements>
````

Your task is to select the judgement from the list above that is the most coherent with both the OBSERVATION and the ACTION.

OBJECTIVE: {intent}
OBSERVATION: {observation}
ACTION: {action}

JUDGES:
{mcq_choices}

Select the judge reasoning that is the most coherent with both the OBSERVATION and the ACTION. Answer in the format of:
OPTION: <one of A, B, C, etc.>
REASON: <reasons for this choice>
""".strip()


def __recover_task_intent(task_info):
    intent  = task_info["intent"]
    images = task_info.get("images", [])
    # Caption the input image, if provided.
    if images is not None and len(images) > 0:
        image_input_caption = ""
        for image_i, image in enumerate(images):
            if image_i == 0:
                image_input_caption += f'Input image {image_i+1}: "{cached_caption_image_fn([image])[0]}"'
            else:
                image_input_caption += f'input image {image_i+1}: "{cached_caption_image_fn([image])[0]}"'
            if len(images) > 1:
                image_input_caption += ", "
        # Update intent to include captions of input images.
        intent = f"{image_input_caption}\nIntent: {intent}"
    return intent


def _mcq_select_best_judge(intent_w_image, state_str, action_str, all_judge_reasons):
    intent = intent_w_image
    try:
        mcq_choices = __format_mcq_choices(all_judge_reasons)
        content = CHOOSE_JUDGE_PROMPT.format(
            intent=intent,
            observation=state_str,
            action=action_str,
            mcq_choices=mcq_choices
        )
        
        completion = create_chat_completion_wrapper(
            messages = [{
                "role": "user",
                "content": content
            }],
            model="gpt-4o",
            temperature=0.7,
            max_tokens=128,
            top_p=0.95,
            num_outputs=1,
        )
        extracted_option = re.search(r"OPTION: (.+)", completion).group(1).strip().capitalize()
        judge_idx = ALPHABET_TO_IDX[extracted_option]
        return all_judge_reasons[judge_idx]
    except Exception as e:
        print(f"Error: {e}")
        completion = ''
    return completion


def _find_best_judge_v2(intent_w_image, state, action_reason, judge_reasonings):
    extracted_reasonings = []
    for reason in judge_reasonings:
        if re.search(r"Thoughts: (.+)", reason) is None:
            continue
        thought = re.search(r"Thoughts: (.+)", reason).group(1)
        extracted_reasonings.append(thought)
    
    selected_v = _mcq_select_best_judge(intent_w_image, state, action_reason, extracted_reasonings)
    return selected_v


def _process_backtrack_action_v2(intent_w_image, prev_state_text: str, action: Action):
    action_clone = copy.deepcopy(action)
    # first get judge
    judges = action.metadata["prev_V_debate_data"]['final_decisions']
    assert len(judges) > 0, f"judges is empty"
    additional_reasoning = _find_best_judge_v2(
        intent_w_image,
        prev_state_text,
        action.raw_prediction,
        judges
    )
    assert additional_reasoning != "", f"additional_reasoning is empty"
    
    # reformat current action
    action_clone.element_id = ''
    if action_clone.action_type == ActionTypes.SCROLL:
        if '```scroll [up]```' in action_clone.raw_prediction:
            backtrack_raw_prediction = BAKCTRACK_SCROLL_DOWN.format(judge_reasoning=additional_reasoning)
        elif '```scroll [down]```' in action_clone.raw_prediction:
            backtrack_raw_prediction = BAKCTRACK_SCROLL_UP.format(judge_reasoning=additional_reasoning)
    elif action_clone.action_type == ActionTypes.GOTO_URL:
        # our manual setting
        real_url = action_clone._backtrack_url
        backtrack_raw_prediction = BAKCTRACK_GOTO_URL.format(judge_reasoning=additional_reasoning, real_url=real_url)
    else:
        backtrack_raw_prediction = BAKCTRACK_BACK.format(judge_reasoning=additional_reasoning)
        action_clone.action_type = ActionTypes.GO_BACK
    
    rephrased = _rephrase_success_estimate_v2(
        intent_w_image,
        prev_state_text,
        backtrack_raw_prediction,
        mode='backtrack'
    )
    action_clone.raw_prediction = rephrased
    return action_clone


def _reformat_action_data_v2(prev_state_text: str, action: Action, task_config: dict):
    intent = __recover_task_intent({
        "intent": task_config['intent'],
        'images': task_config.get('images', [])
    })

    if getattr(action, '_direction', '') == "backtrack":
        return _process_backtrack_action_v2(intent, prev_state_text, action)

    if 'prev_V' not in action.metadata:
        # first action
        return action

    action_clone = copy.deepcopy(action)
    judges = action.metadata["prev_V_debate_data"]['final_decisions']
    additional_reasoning = _find_best_judge_v2(
        intent,
        prev_state_text,
        action.raw_prediction,
        judges
    )
    assert additional_reasoning != "", f"additional_reasoning is empty"

    forward_new_prediction = FORWARD_PROMPT.format(
        judge_reasoning=additional_reasoning,
        raw_action_prediction=action.raw_prediction
    )
    rephrased = _rephrase_success_estimate_v2(
        intent,
        prev_state_text,
        forward_new_prediction,
        mode='forward'
    )

    action_clone.raw_prediction = rephrased
    return action_clone


def _get_dtype(data):
    if isinstance(data, dict):
        prev_last_dtype = "state"
    elif isinstance(data, Node):
        prev_last_dtype = "state"
    elif isinstance(data, Action):
        prev_last_dtype = "action"
    else:
        raise Exception(f"Unknown data type {type(data)}")
    return prev_last_dtype


def merge_single_traversals(single_consecutive_traversal, task_config=None, dry_run=False):
    ### step 1. concate all trajectories
    all_concat_data = single_consecutive_traversal[0]
    for partial_traj in single_consecutive_traversal[1:]:
        prev_last_data = all_concat_data[-1]
        prev_last_dtype = _get_dtype(prev_last_data)
        
        curr_first_dtype = _get_dtype(partial_traj[0])
        if curr_first_dtype != "state":
            raise Exception(f"First data type is not state: {curr_first_dtype}")
        
        # easy if we ended with an action
        if prev_last_dtype == "action":
            # get in!
            all_concat_data.extend(partial_traj)
        else:
            # remove prev state
            all_concat_data = all_concat_data[:-1]
            all_concat_data.extend(partial_traj)

    last_data_dtype = _get_dtype(all_concat_data[-1])
    if last_data_dtype == "state":
        # remove the last state
        all_concat_data = all_concat_data[:-1]

    ### step 1.5 add additional metadata
    prev_action = None
    for data in all_concat_data:
        if isinstance(data, Action):
            if prev_action is not None:
                curr_v_debate = prev_action.metadata["next_V_debate_data"]
                curr_v = prev_action.metadata["V_next"]
                data.metadata["prev_V_debate_data"] = curr_v_debate
                data.metadata["prev_V"] = curr_v
            prev_action = data

    ### step 2. convert all backtrack actions
    backtracked_data = []
    for idx, data in enumerate(all_concat_data):
        if isinstance(data, tuple):
            data, _ = data
        if isinstance(data, Action):
            if idx == 0:
                prev_data = None
            else:
                prev_data = all_concat_data[idx-1]
            
            if not dry_run:
                # data = _reformat_action_data(prev_data, data, task_config)
                # prev data should be dict
                prev_state_str = prev_data['observation']['text']
                data = _reformat_action_data_v2(prev_state_str, data, task_config)
        elif isinstance(data, Node):
            try:
                data = data.trajectory[-1]
            except:
                print('encountering terminal state before last action')
                print('error')
                break
        backtracked_data.append(data)
    return backtracked_data


def _traj_num_consecutive_backtrack(trajectory):
    consec_nums = []
    num_consecutive = 0
    is_prev_backtrack = False
    for data in trajectory:
        if isinstance(data, Action):
            direction = getattr(data, '_direction', '')
            if direction == 'backtrack':
                if is_prev_backtrack:
                    num_consecutive += 1
                else:
                    num_consecutive = 1
                is_prev_backtrack = True
            else:
                is_prev_backtrack = False
                consec_nums.append(num_consecutive)
                num_consecutive = 0
    consec_nums.append(num_consecutive)
    return max(consec_nums)


BACKTRACK_KWD = "We should take a step back"
def _traj_has_backtrack(traj_list):
    for data in traj_list:
        if isinstance(data, Action):
            if BACKTRACK_KWD in data.raw_prediction:
                return True
    return False


def _filter_traj(traj: list):
    # make sure last actoin id can be found
    action_element_id = traj[-1].element_id
    if action_element_id == '':
        return False
    element_matching_text = f"[{action_element_id}]"
    observation = traj[-2]['observation']['text']
    if element_matching_text not in observation:
        return True  # remove

    # check if the special field _remove_from_training is there
    action = traj[-1]
    if getattr(action, '_remove_from_training', False):  # set by the preview_edit_tree_trajectory.py
        print('removing due to _remove_from_training')
        return True
    return False # keep


def _remove_value_before_action(action):
    prediction_str = action.raw_prediction
    if "Let's see what we have got" in prediction_str:
        # remove the first sentence
        sents = prediction_str.split("\n")
        new_sents = sents[1:]
        new_prediction_str = "\n".join(new_sents)
        action.raw_prediction = new_prediction_str
    return action


def _remove_backtrack_to_normal_action(traj):
    new_traj = []
    for data in traj:
        if isinstance(data, Action):
            copied_data = copy.deepcopy(data)
            if getattr(copied_data, '_direction', '') == "backtrack":
                # skip checking following, which is patched manually
                backtrack_url = getattr(copied_data, '_backtrack_url', '')
                if backtrack_url == 'http://onestopmarket.com/home-kitchen/heating-cooling-air-quality.html?product_list_order=price':
                    directly_correct_action = _remove_value_before_action(copied_data)
                    new_traj.append(directly_correct_action)
                    continue
                if backtrack_url == 'http://wikipedia.org/search?content=wikipedia_en_all_maxi_2022-05&pattern=Major+commercial+airport+in+Washington+state':
                    copied_data._n_action_reversed = 1

                n_action_reversed = copied_data._n_action_reversed
                if n_action_reversed > 1:
                    # assert n_action_reversed == 1, f"n_action_reversed is not 1: {n_action_reversed}"
                    # pop all the way until the state has the same url as this one
                    bactrack_url = copied_data._backtrack_url
                    while True:
                        prev_data = new_traj[-1]
                        if isinstance(prev_data, dict):
                            converted_url = map_url_to_real(prev_data['url'])
                            if converted_url == bactrack_url:
                                new_traj.pop()
                                break
                        new_traj.pop()
                else:
                    # was (a,s,a to bracktrack,s,backtrack,s,a), pop
                    # skip this and pop s,a,s
                    new_traj.pop()
                    new_traj.pop()
                    new_traj.pop()
            else:
                directly_correct_action = _remove_value_before_action(copied_data)
                new_traj.append(directly_correct_action)
        else:
            new_traj.append(data)
    return new_traj


def _process_exploratory_learning_data(env_name, result_dir, output_dir):
    args = argparse.Namespace(
        instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final_norefl_noicl_tree.json",
    )

    prompt_constructor = ExploratoryCoTPolicyPConstructor(
        instruction_path=args.instruction_path,
        lm_config=llm_config,
        tokenizer=llm_tokenizer
    )

    successful_data = find_successful_traj_w_trees(result_dir)

    num_passed = 0
    num_failed = 0
    total_length = {}
    for i in range(len(successful_data)):
        print('processing', i)
        try:
            all_traversals = process_single_traj(successful_data[i]["traj"], successful_data[i]["trees"])
            num_passed += 1
            curr_len = 0
            for traversal in all_traversals:
                curr_len += len(traversal)

            if curr_len not in total_length:
                total_length[curr_len] = 0
            total_length[curr_len] += 1
        except Exception as e:
            num_failed += 1
            print(traceback.format_exc())
            print('============')

    
    task_config_base = DSET_NAME_TO_FOLDER[env_name]
    num_passed = 0
    num_failed = 0
    num_skipped = 0
    total_length = {}
    all_formatted_traj = []
    all_formatted_tids = []
    pbar = tqdm(total=len(successful_data[:]))
    for i in range(len(successful_data[:])):
        print('processing', i)
        try:
            tid = successful_data[i]['task_id']
            config_file_path = f"{task_config_base}/{tid}.json"
            with open(config_file_path, 'r') as f:
                task_config = json.load(f)

            all_traversals = process_single_traj(successful_data[i]["traj"], successful_data[i]["trees"])
            single_traversal = merge_single_traversals(all_traversals, task_config, dry_run=False)  # use dry_run=True to check errors first
            n_backtrack = _traj_num_consecutive_backtrack(single_traversal)
            print(f"idx={i}, {n_backtrack=}")
            if len(single_traversal) > 35:
                print('length > 35, skipping')
                num_skipped += 1
                pbar.update(1)
                continue

            num_passed += 1
            curr_len = len(single_traversal)
            if curr_len not in total_length:
                total_length[curr_len] = 0
            total_length[curr_len] += 1

            all_formatted_traj.append(single_traversal)
            all_formatted_tids.append(tid)
        except Exception as e:
            print('processing error at', i)
            print(traceback.format_exc())
            print('============')
            num_failed += 1
        pbar.update(1)
    save_llm_cache()

    print('collected', len(all_formatted_traj))
    for k, v in total_length.items():
        print(f"length={k}, {v=}")

    ### save
    tree_traj_save_path = os.path.join(output_dir, f"tmp_{env_name}_tree.pkl.xz")

    truncated_formatted_trajs = []
    for traj in all_formatted_traj:
        truncated_traj = traj[:100] # no truncation
        truncated_formatted_trajs.append(truncated_traj)

    with lzma.open(tree_traj_save_path, "wb") as fwrite:
        pickle.dump(truncated_formatted_trajs, fwrite)

    # also save the tisd
    tree_tid_save_path = os.path.join(output_dir, f"tmp_{env_name}_tree_tids.txt")
    all_tids_str = ",".join([str(x) for x in all_formatted_tids])
    with open(tree_tid_save_path, "w") as fwrite:
        fwrite.write(all_tids_str)

    ### subsampling
    rng = random.Random(42)

    # lets make max traj to 30
    NUM_TRAJ = 30
    num_has_backtrack = 0
    _has_backtrack_trajs = []
    _has_backtrack_tids = []
    _non_backtrack_trajs = []
    _non_backtrack_tids = []

    for traj, tid in zip(truncated_formatted_trajs, all_formatted_tids):
        if _traj_has_backtrack(traj):
            _has_backtrack_trajs.append(traj)
            _has_backtrack_tids.append(tid)
            num_has_backtrack += 1
        else:
            _non_backtrack_trajs.append(traj)
            _non_backtrack_tids.append(tid)

    rng.shuffle(_non_backtrack_trajs)
    # truncate non backtrack such that total length is 30
    num_non_backtrack_to_keep = NUM_TRAJ - num_has_backtrack
    _non_backtrack_trajs = _non_backtrack_trajs[:num_non_backtrack_to_keep]
    _non_backtrack_tids = _non_backtrack_tids[:num_non_backtrack_to_keep]

    # combine
    rebalanced_filtered_trainable_tree_chats = _has_backtrack_trajs + _non_backtrack_trajs
    rebalanced_filtered_trainable_tids = _has_backtrack_tids + _non_backtrack_tids
    percent_has_backtrack = num_has_backtrack / len(rebalanced_filtered_trainable_tree_chats)
    print(f"num has backtrack: {num_has_backtrack} out of {len(rebalanced_filtered_trainable_tree_chats)}, percentage {percent_has_backtrack*100.0:.2f}%")

    # finally, convert to chat data
    modality = "text"
    raw_trainable_tree_chats = []
    raw_kept_tids = []
    num_filtered = 0
    pbar = tqdm(total=len(rebalanced_filtered_trainable_tree_chats))
    for tid, traj in zip(rebalanced_filtered_trainable_tids, rebalanced_filtered_trainable_tree_chats):
        ### get all separated trajectory
        # (s,a), (s,a,s,a), ...
        end_idx = 2
        while end_idx < len(traj)+1:
            partial_traj = copy.deepcopy(traj[:end_idx])
            # truncation
            if len(partial_traj) > 16:
                # cut to max length of 16
                print('truncating partial traj in', tid)
                partial_traj = partial_traj[:4] + partial_traj[-12:]
            if _filter_traj(partial_traj):
                end_idx += 2
                continue

            trainable_chat = get_single_training_data_from_trajectory(partial_traj, tid, env_name, prompt_constructor, modality)
            if _filter_train_data(trainable_chat['messages']):
                num_filtered += 1
            else:
                # good
                print('post processing', tid)
                # no rephrase since during editing, this should be taken care of
                trainable_chat = _trainable_chat_postprocessing(trainable_chat, no_rephrase=True)
                raw_trainable_tree_chats.append(trainable_chat)
                raw_kept_tids.append(tid)
            end_idx += 2
        pbar.update(1)
    
    print('in total length', len(set(raw_kept_tids)))
    print(set(raw_kept_tids))

    # save data
    str_date = datetime.now().strftime("%m%d")
    final_tree_train_path = os.path.join(output_dir, f"{env_name}_puretext_tree_{str_date}.jsonl")
    with jsonlines.open(final_tree_train_path, "w") as fwrite:
        fwrite.write_all(raw_trainable_tree_chats)
    print(f"saved {len(raw_trainable_tree_chats)} chats to {final_tree_train_path}")

    final_tree_train_tid_path = os.path.join(output_dir, f"{env_name}_puretext_tree_{str_date}_tids.txt")
    with open(final_tree_train_tid_path, "w") as fwrite:
        fwrite.write(",".join([str(x) for x in raw_kept_tids]))
    print(f"saved {len(raw_kept_tids)} tids to {final_tree_train_tid_path}")
    return rebalanced_filtered_trainable_tids, rebalanced_filtered_trainable_tree_chats


def _process_imitation_learning_data(env_name, trajectories, tids, output_dir):
    args = argparse.Namespace(
        instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final_norefl_noicl.json",
    )
    prompt_constructor = CoTPolicyPConstructor(
        instruction_path=args.instruction_path,
        lm_config=llm_config,
        tokenizer=llm_tokenizer
    )

    directly_correct_tids = []
    directly_correct_chats = []
    for tid, traj in zip(tids, trajectories):
        flattened_traj = _remove_backtrack_to_normal_action(traj)
        directly_correct_tids.append(tid)
        directly_correct_chats.append(flattened_traj)

    # process to jsonl training data
    modality = "text"
    raw_trainable_flat_chats = []
    raw_kept_tids = []
    num_filtered = 0
    pbar = tqdm(total=len(directly_correct_chats))
    for tid, traj in zip(directly_correct_tids, directly_correct_chats):
        ### get all separated trajectory
        # (s,a), (s,a,s,a), ...|
        end_idx = 2
        while end_idx < len(traj)+1:
            partial_traj = copy.deepcopy(traj[:end_idx])
            # truncation
            if len(partial_traj) > 16:
                # cut to max length of 16
                print('truncating partial traj in', tid)
                partial_traj = partial_traj[:4] + partial_traj[-12:]
            if _filter_traj(partial_traj):
                end_idx += 2
                continue

            trainable_chat = get_single_training_data_from_trajectory(partial_traj, tid, env_name, prompt_constructor, modality)
            if _filter_train_data(trainable_chat['messages']):
                num_filtered += 1
            else:
                # good
                print('post processing', tid)
                # no rephrase since during editing, this should be taken care of
                trainable_chat = _trainable_chat_postprocessing(trainable_chat, no_rephrase=True)
                raw_trainable_flat_chats.append(trainable_chat)
                raw_kept_tids.append(tid)
            end_idx += 2
        pbar.update(1)


    str_date = datetime.now().strftime("%m%d")
    final_flat_train_path = os.path.join(output_dir, f"{env_name}_puretext_flat_{str_date}.jsonl")
    with jsonlines.open(final_flat_train_path, "w") as fwrite:
        fwrite.write_all(raw_trainable_flat_chats)
    print(f"saved {len(raw_trainable_flat_chats)} chats to {final_flat_train_path}")

    final_flat_train_tid_path = os.path.join(output_dir, f"{env_name}_puretext_flat_{str_date}_tids.txt")
    with open(final_flat_train_tid_path, "w") as fwrite:
        fwrite.write(",".join([str(x) for x in raw_kept_tids]))
    print(f"saved {len(raw_kept_tids)} (unique = {len(set(raw_kept_tids))}) tids to {final_flat_train_tid_path}")
    return


def main(args):
    env_name = args.env_name
    result_dir = args.result_dir
    output_dir = args.output_dir

    # first process exploratory learning data
    trajs, tids = _process_exploratory_learning_data(env_name, result_dir, output_dir)
    # convert processed data back to imitation learning data for comparison
    _process_imitation_learning_data(env_name, trajs, tids, output_dir)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="classifields")
    parser.add_argument("--result_dir", type=str, help="directory containing RMCTS_mad agent eval results")
    parser.add_argument("--output_dir", type=str, help="directory to save the processed data")
    args = parser.parse_args()

    main(args)

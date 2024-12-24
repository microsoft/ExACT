import copy
import logging
import math
import os
import time
import lzma
import pickle
from src.logging import atime_it, time_it
from typing import Any, Optional
from PIL import Image
from dataclasses import dataclass, field
from collections import defaultdict
from graphviz import Digraph

from agent.prompts import PromptConstructor
from src.agentic.policy import MCoTPolicyPConstructor_OLD
from browser_env import Trajectory, ActionParsingError, ActionTypes
from src.envs.actions import (
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_stop_action,
)
from src.helper_functions import get_action_description
from src.agentic.value_function import ValueFunction, DirectCoTValueFunction
from src.envs.browser import FastCachedwActionMatchingBrowserEnv
from src.envs.actions import ActionTypes, Action
from src.llms import lm_config
from src.llms.utils import call_llm, is_vlm
from src.agent.base_agent import FastAgent
from src.agent.agent_args import AgentArguments
from src.agent.utils import SoftBudgetTracker


logger = logging.getLogger('mcts_agent')


@dataclass
class Node:
    env: FastCachedwActionMatchingBrowserEnv
    trajectory: Trajectory
    action_trajectory: list[Action]
    action_trajectory_str: list[str]  # this need to be immutable so that state hash is immutable
    action_set_tag: str
    value: float
    children: dict[Action, 'Node']
    ## helper info
    _additional_info: dict[str, Any] = field(default_factory=dict)
    _lazy_initialized: bool = True  # to save time, the next state may NOT always be initialized
    _need_evaluation: bool = True
    Ns: int = 0
    depth: int = 0
    is_root: bool = False
    is_terminal: bool = False

    def _to_string_rep(self) -> str:
        return ' -> '.join(self.action_trajectory_str)

    def _get_all_child_actions(self) -> list[tuple[Action, 'Node']]:
        # recursively get all children
        child_state_actions = []
        for a, s in self.children.items():
            child_state_actions.append((a, s))
            child_actions = s._get_all_child_actions()
            child_state_actions.extend(child_actions)
        return child_state_actions

    def get_all_children_str(self, Q: dict = None) -> list[str]:
        all_children = []
        all_a_s = self._get_all_child_actions()
        for a, s in all_a_s:
            depth = s.depth
            # #skip if value is zero = may not have gone to the _simulation step
            # if s.value == 0:
            #     continue
            prefix = "..." * depth + "|_"

            s_hashed = s._to_string_rep()
            q = 'nan'
            if Q is not None:
                q = Q.get(s_hashed, {}).get(a, 'nan')
            all_children.append(
                f'{prefix} depth={depth}: {a["raw_prediction"]} (Ns={s.Ns}, q={q}, v={s.value}, p={a["prob"]})'
            )
        return all_children

    def __hash__(self):
        return hash(self._to_string_rep())


def break_long_string(long_str: str):
    limit = 140
    if len(long_str) < limit:
        return long_str
    else:
        return long_str[:limit] + "\l" + break_long_string(long_str[limit:])


class MCTSRenderHelper:
    """helps render the MCTS search tree for debugging/analysis
    """
    def __init__(self, save_dir, tmp_image_save_dir) -> None:
        self.save_dir = save_dir
        self.tmp_image_save_dir = tmp_image_save_dir

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(tmp_image_save_dir, exist_ok=True)
        self._tmp_images_created = []
        return

    def create_graphviz_node(self, state: Node, node_idx: int, graph):
        image_to_display = state._additional_info['last_screenshots'][-1]
        curr_time = time.strftime("%Y%m%d-%H%M%S")
        curr_pid = os.getpid()  # to avoid parallel run conflicts
        save_file_name = os.path.join(os.path.abspath(self.tmp_image_save_dir), f"{curr_time}_node_{node_idx}_p{curr_pid}.png")
        image_to_display.save(save_file_name)
        self._tmp_images_created.append(save_file_name)

        graph.node(f'{node_idx}', label=f"v={state.value}, Ns={state.Ns}", image=save_file_name)
        return

    def _dfs_nodes(self, state: Node, graph: Digraph, Q: dict, seen_nodes: dict):
        node_hash = state._to_string_rep()
        assert node_hash not in seen_nodes, f"Node {node_hash} already seen"
        curr_node_idx = len(seen_nodes) + 1
        seen_nodes[node_hash] = curr_node_idx
        self.create_graphviz_node(state, curr_node_idx, graph)

        for a, next_s in state.children.items():
            child_hash = next_s._to_string_rep()
            qsa = Q[node_hash].get(a, 0.0)

            if a["action_type"] in [ActionTypes.STOP, ActionTypes.NONE]:
                # special node, no next state image
                curr_node_idx = len(seen_nodes) + 1
                seen_nodes[child_hash] = curr_node_idx
                graph.node(f'{curr_node_idx}', label=f"stop")
            elif next_s._lazy_initialized:
                # next state image not rendered
                curr_node_idx = len(seen_nodes) + 1
                seen_nodes[child_hash] = curr_node_idx
                graph.node(f'{curr_node_idx}', label=f"lazy init")
            elif child_hash not in seen_nodes:
                self._dfs_nodes(next_s, graph, Q, seen_nodes)
            # format label to display
            a_prob = getattr(a, 'prob', 0.0)
            value_str = f"Q={qsa:.2f}, V next={next_s.value}, N next={next_s.Ns}, p(Action)={a_prob:.2f}"
            act_str = f"Action={next_s.action_trajectory_str[-1]}".strip().replace('\n]', '↵]')
            reason_str = f"Reasoning={a.raw_prediction}".strip().replace('\n]', '↵]')
            reason_str = break_long_string(reason_str).strip().replace('\n', '\l')
            label = fr"{value_str}\l" + fr"{act_str}\l" + fr"{reason_str}\l"
            graph.edge(f'{seen_nodes[node_hash]}', f'{seen_nodes[child_hash]}', label=label)
        return

    def get_graphview(self, root_node: Node, Q: dict):
        graph = Digraph(node_attr={'shape': 'box'})
        graph.body.insert(0, '\tnodesep="9.0"\n')
        graph.body.insert(0, '\tranksep="5.0"\n')
        seen_nodes = {}

        self._dfs_nodes(root_node, graph, Q, seen_nodes)
        return graph

    def cleanup(self):
        for img_file in self._tmp_images_created:
            os.remove(img_file)
        self._tmp_images_created = []
        return

    @time_it
    def render(self, root_node: Node, Q: dict):
        try:
            graph = self.get_graphview(root_node, Q)

            curr_time = time.strftime("%Y%m%d-%H%M%S")
            graph.render(os.path.join(self.save_dir, curr_time), format='pdf')
        except Exception as e:
            logger.error(e, exc_info=True)
        
        self.cleanup()
        return


class MCTSwDBTRenderHelper(MCTSRenderHelper):
    """helps render the MCTS search tree + debate based value function for debugging/analysis
    """
    def _dfs_nodes(self, state: Node, graph: Digraph, Q: dict, seen_nodes: dict):
        node_hash = state._to_string_rep()
        assert node_hash not in seen_nodes, f"Node {node_hash} already seen"
        curr_node_idx = len(seen_nodes) + 1
        seen_nodes[node_hash] = curr_node_idx
        self.create_graphviz_node(state, curr_node_idx, graph)

        for a, next_s in state.children.items():
            child_hash = next_s._to_string_rep()
            qsa = Q[node_hash].get(a, 0.0)

            if a["action_type"] in [ActionTypes.STOP, ActionTypes.NONE]:
                # special node, no next state image
                curr_node_idx = len(seen_nodes) + 1
                seen_nodes[child_hash] = curr_node_idx
                graph.node(f'{curr_node_idx}', label=f"stop")
            elif next_s._lazy_initialized:
                # next state image not rendered
                curr_node_idx = len(seen_nodes) + 1
                seen_nodes[child_hash] = curr_node_idx
                graph.node(f'{curr_node_idx}', label=f"lazy init")
            elif child_hash not in seen_nodes:
                self._dfs_nodes(next_s, graph, Q, seen_nodes)
            # format label to display
            a_prob = getattr(a, 'prob', 0.0)
            value_str = f"Q={qsa:.2f}, V next={next_s.value}, N next={next_s.Ns}, p(Action)={a_prob:.2f}"
            debate_str = "V next debate=\l"
            debate_data = next_s._additional_info.get('debate_data', {})
            if len(debate_data) != 0:
                debate_support = break_long_string(debate_data.get('supporting_reasons', "")).strip()
                debate_oppose = break_long_string(debate_data.get('opposing_reasons', "")).strip()
                debate_str += f"V next supporting={debate_support}\l"
                debate_str += f"V next opposing={debate_oppose}\l"

            act_str = f"Action={next_s.action_trajectory_str[-1]}".strip().replace('\n]', '↵]')
            reason_str = f"Reasoning={a.raw_prediction}".strip().replace('\n]', '↵]')
            reason_str = break_long_string(reason_str).strip().replace('\n', '\l')
            label = fr"{value_str}\l" + fr"{debate_str}\l======\l" + fr"{act_str}\l" + fr"{reason_str}\l"
            graph.edge(f'{seen_nodes[node_hash]}', f'{seen_nodes[child_hash]}', label=label)
        return

class MCTSAgent(FastAgent):
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        value_function: ValueFunction,
        captioning_fn = None,
        early_stop_fn = None
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.value_function = value_function
        self.captioning_fn = captioning_fn
        self.early_stop_fn = early_stop_fn

        # Check if the model is multimodal.
        if is_vlm(self.lm_config) and prompt_constructor.is_multimodal:
            self.multimodal_inputs = True
            logger.info("Using multimodal input in prompt.")
        else:
            self.multimodal_inputs = False
            logger.info("Model is not multimodal.")
        
        ## MCTS stats
        self.Ns: dict = {}  # saves compute
        self.Nsa: dict = {}
        self.Q: dict = {}
        self.P: dict = {}
        self.branching_factor = 5
        self.value_function_model = "gpt-4o-mini"
        # utility
        self.Q_0 = 0.0
        self.cpuct = 1.0
        self.found_success_trajectory = False
        self.best_action_cache: list[Action] = []
        return

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag
        return

    @time_it
    def _gen_next_actions(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
    ) -> list[Action]:
        logger.info(f"Generating next actions")
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        observation_metadata = state_info['info']['observation_metadata']

        # Create page screenshot image for multimodal models.
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["image"]
            page_screenshot_img = Image.fromarray(
                page_screenshot_arr
            )  # size = (viewport_width, viewport_width)

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                logger.warn(
                    "WARNING: Input image provided but no image captioner available."
                )

        if self.multimodal_inputs:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, page_screenshot_img, images, meta_data
            )
        else:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, meta_data
            )
        lm_config = self.lm_config
        n = 0
        while True:
            responses = call_llm(
                lm_config,
                prompt,
                num_outputs=max(self.branching_factor * 2, 20)
            )
            if type(responses) == str:
                responses = [responses]
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            n += 1
            all_actions = {}
            all_actions_count = {}

            for response in responses:
                response = f"{force_prefix}{response}"
                try:
                    parsed_response = self.prompt_constructor.extract_action(
                        response
                    )
                    if self.action_set_tag == "id_accessibility_tree":
                        action = create_id_based_action(parsed_response)
                    elif self.action_set_tag == "playwright":
                        action = create_playwright_action(parsed_response)
                    elif self.action_set_tag == "som":
                        action = create_id_based_action(parsed_response)
                    else:
                        raise ValueError(
                            f"Unknown action type {self.action_set_tag}"
                        )
                    action_hash = get_action_description(
                        action,
                        observation_metadata=observation_metadata,
                        action_set_tag=self.action_set_tag,
                        prompt_constructor=None
                    )
                    if action_hash in all_actions:
                        all_actions_count[action_hash] += 1
                    else:
                        # only add unique actions to all_actions
                        action['metadata']['obs_metadata'] = copy.deepcopy(observation_metadata)

                        all_actions_count[action_hash] = 1
                        action["raw_prediction"] = response
                        all_actions[action_hash] = action
                except ActionParsingError as e:
                    continue
            
            # If any valid action is found, break.
            if len(all_actions) > 0:
                break
            else:
                # If no valid action is found, retry.
                # If the number of retries exceeds the maximum, return a None action.
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    action["prob"] = 0.0
                    return [action]
                
        # Find top branching_factor actions.
        top_action_hashes = sorted(
            all_actions_count,
            key=all_actions_count.get, reverse=True
        )[:self.branching_factor]
        top_action_count = sum([all_actions_count[action_hash] for action_hash in top_action_hashes])
        updated_actions = []
        for action_hash in top_action_hashes:
            a = all_actions[action_hash]
            a['prob'] = all_actions_count[action_hash] / top_action_count
            updated_actions.append(a)

        ### check if element id is found on the page
        logger.debug(f"Last turn prompt:\n")
        if isinstance(prompt[-1]['content'], list):
            # multiple messages
            for msg in prompt[-1]['content']:
                if 'text' in msg:
                    logger.debug(f"text:\n{msg['text']}")
                elif 'image_url' in msg:
                    logger.debug(f"image_url: {msg['image_url']['url'][:50]}...(truncated)")
        else:
            logger.debug(f"{prompt[-1]['content']}")
        logger.debug(f"LLM Top actions:")
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs_text = state_info["observation"]["text"]
        for a in updated_actions:
            logger.debug(f"  {a['raw_prediction']}")
            # check if is valid
            action_element_id = a['element_id']
            if action_element_id == "":
                continue
            if action_element_id in obs_text:
                logger.debug(f"  [{action_element_id}] is found on the page!")
            else:
                logger.debug(f"  [{action_element_id}] is NOT found on the page!")
        return updated_actions

    def _expansion(self, state: Node):
        intent = state._additional_info['intent']
        meta_data = state._additional_info['meta_data']
        images = state._additional_info.get('images', None)
        new_actions = self._gen_next_actions(
            state.trajectory,
            intent,
            meta_data,
            images=images
        )
        prior = {}
        for a in new_actions:
            ### initialize new state/node
            is_terminal = False
            need_evaluation = True
            if a["action_type"] in [ActionTypes.STOP, ActionTypes.NONE]:
                is_terminal = True
            if a["action_type"] in [ActionTypes.NONE]:
                need_evaluation = False  # will just return 0.0

            act_desc = get_action_description(
                a,
                observation_metadata=state._additional_info["observation_metadata"],
                action_set_tag=self.action_set_tag,
                prompt_constructor=None
            )
            a.metadata["Q"] = self.Q_0
            a.metadata["Nsa"] = 0
            a.metadata["P"] = a["prob"]
            a.metadata["V_next"] = 0.0

            new_state = Node(
                env=state.env,
                trajectory=state.trajectory,  # these will be updated after astep
                action_trajectory=copy.deepcopy(state.action_trajectory) + [a],
                action_trajectory_str=state.action_trajectory_str + [act_desc],
                action_set_tag=self.action_set_tag,
                value=0.0,
                children={},
                _additional_info=copy.deepcopy(state._additional_info),  # these will be fully updated after astep in _get_next_state
                depth=state.depth + 1,
                _lazy_initialized=True,
                _need_evaluation=need_evaluation,
                is_terminal=is_terminal
            )
            new_state._additional_info['meta_data']['action_history'].append(act_desc)
            state.children[a] = new_state
            prior[a] = a['prob']

        ### update Q, Ns, Nsa
        hashable_state = state._to_string_rep()
        logger.debug(f"expanded state: {hashable_state} with len={len(prior)} child actions")
        assert hashable_state not in self.Ns, f"state {hashable_state} already expanded"
        self.Ns[hashable_state] = 0
        self.Nsa[hashable_state] = defaultdict(lambda: 0.0)
        self.Q[hashable_state] = defaultdict(lambda: self.Q_0)  # 0.0 for Q[s][new_a]
        self.P[hashable_state] = prior  # distribution given a (hashed_s, a)
        return

    async def _simulation(self, state: Node) -> float:
        ### return value by (optionally) doing roll out
        logger.info("Simulating state value")
        if state.is_root:
            return 0.0
        
        v_model = self.value_function_model
        intent = state._additional_info['intent']
        images = state._additional_info.get('images', [])  # task intent images
        try:
            # evaluate current state
            # last_screenshots = screenshots SINCE the root state
            last_screenshots = state._additional_info["last_screenshots"]  # [-4:] is now done inside value function
            init_screenshot = Image.fromarray(state.trajectory[0]["observation"]["image"])
            if self.value_function_model in ["gpt4o"]:
                v = self.value_function.evaluate_success(
                    screenshots=last_screenshots,
                    actions=copy.deepcopy(state.action_trajectory_str),
                    current_url=state.env.page.url,
                    last_reasoning="",
                    intent=intent,
                    models=["gpt-4o-2024-05-13"],
                    init_screenshot=init_screenshot,
                    intent_images=images if len(images) > 0 else None
                )
            elif self.value_function_model in ["gpt-4o", "gpt-4o-mini", "OpenGVLab/InternVL2-Llama3-76B"]:
                v = self.value_function.evaluate_success(
                    screenshots=last_screenshots,
                    actions=copy.deepcopy(state.action_trajectory_str),
                    current_url=state.env.page.url,
                    last_reasoning="",
                    intent=intent,
                    models=[v_model],
                    init_screenshot=init_screenshot,
                    intent_images=images if len(images) > 0 else None
                )
            else:
                raise NotImplementedError(f"Value function model {v_model} not implemented")
        except Exception as e:
            logger.error(e, exc_info=True)
            v = 0.0
        
        state.value = v
        state.Ns += 1
        state._need_evaluation = False
        return v

    async def _get_next_state(self, state: Node, action: Action) -> Node:
        next_state = state.children[action]

        new_trajectory = copy.deepcopy(state.trajectory)
        new_trajectory.append(action)
        new_additional_info = copy.deepcopy(next_state._additional_info)

        if action["action_type"] == ActionTypes.STOP:
            next_state.is_terminal = True
            next_state.trajectory = new_trajectory
            next_state._additional_info = new_additional_info
            next_state._lazy_initialized = False
            return next_state
        elif action["action_type"] == ActionTypes.NONE:  # invalid action
            next_state.is_terminal = True
            next_state.trajectory = new_trajectory
            next_state._additional_info = new_additional_info
            next_state._lazy_initialized = False
            return next_state
        
        # FastCachedwActionMatchingBrowserEnv will alter the action
        obs, _, terminated, _, info = await state.env.astep(copy.deepcopy(action))

        new_trajectory.append({"observation": obs, "info": info, "url": state.env.page.url})
        if terminated:
            new_trajectory.append(create_stop_action(""))
            next_state.is_terminal = True

        if self.early_stop_fn is not None:
            stop_flag, _ = self.early_stop_fn(new_trajectory)
            if stop_flag:
                next_state.is_terminal = True
                next_state._need_evaluation = False  # don't eval, we are dead

        new_additional_info['observation_metadata'] = info['observation_metadata']
        new_additional_info['last_screenshots'].append(Image.fromarray(obs["image"]))

        next_state.trajectory = new_trajectory
        next_state._additional_info = new_additional_info
        next_state._lazy_initialized = False
        return next_state
    
    async def search(self, state: Node):
        """perform one iteration of MCTS: selection, expansion, simulation, backpropagation
        """
        hashable_state = state._to_string_rep()
        
        v = 0.0
        # if this leaf node is terminal, return the value
        if state.is_terminal:
            # terminal node
            logger.debug(f"terminal state: {hashable_state}")
            if state._need_evaluation:
                await self._simulation(state)
            return state.value
        elif state.value == 1.0:
            best_action_strs = [a.to_short_str() for a in self.best_action_cache]
            best_action_str = "\n--->\n".join(best_action_strs)
            logger.info(f"found task finishing trajectory: {best_action_str}")
            self.found_success_trajectory = True
            return state.value
        elif len(state.children) == 0:
            # selected leaf node, expand and simulate (for backprop below)
            self._expansion(state)
            await self._simulation(state)
            return state.value
        
        
        ##### Selection
        # existing, continue selection
        # go next state by picking best according to U(s,a)
        best_uct = -float('inf')
        best_action = None
        for a in state.children.keys():
            Ns = self.Ns[hashable_state]
            qsa = self.Q[hashable_state][a]
            p = self.P[hashable_state][a]
            nsa = self.Nsa[hashable_state][a]
            if Ns == 0:  # first time visit
                uct = qsa + self.cpuct * p
            else:
                uct = qsa + self.cpuct * p * math.sqrt(Ns) / (1 + nsa)
            
            if uct > best_uct:
                best_uct = uct
                best_action = a
                logger.debug(f"updating best action: {best_action.raw_prediction}")
                logger.debug(f"uct={uct} (with {Ns=}, {nsa=}, {qsa=}, {p=})")
        logger.debug(f"selected best action: {best_action.raw_prediction}")
        
        # transition and update that state's metadata
        self.best_action_cache.append(best_action)
        next_state = await self._get_next_state(state, best_action)
        
        ##### Expansion and Simulation
        # 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
        # 2. if leaf, we will expand it and return the value for backpropagation
        v = await self.search(next_state)


        ##### Backpropagation
        # update stats
        # add in new estimate and average
        self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][best_action] + v) / (self.Nsa[hashable_state][best_action] + 1)
        logger.debug(f"backpropagating value {v} to get Q[{hashable_state}][{best_action.raw_prediction}]={self.Q[hashable_state][best_action]}")
        self.Nsa[hashable_state][best_action] += 1
        self.Ns[hashable_state] += 1
        state.Ns += 1

        # update metadata in action
        best_action.metadata["Q"] = self.Q[hashable_state][best_action]
        best_action.metadata["Nsa"] = self.Nsa[hashable_state][best_action]
        best_action.metadata["V_next"] = next_state.value

        if v == 1.0:
            best_action_strs = [a.to_short_str() for a in self.best_action_cache]
            best_action_str = "\n--->\n".join(best_action_strs)
            logger.info(f"found task finishing trajectory: {best_action_str}")
            self.found_success_trajectory = True
        return v

    async def reset_env(self, root_node: Node, config_file: str, action_history: list[Action]):
        env = root_node.env

        # Reset environment to prepare for next action.
        obs, info = await env.areset(options={"config_file": config_file})
        new_trajectory = [{"observation": obs, "info": info, "url": env.page.url}]
        # Take all the previous actions to get back to the current state.
        for a_hist in action_history:
            new_trajectory.append(a_hist)
            obs, _, _, _, info = await env.astep(a_hist)
            new_trajectory.append({"observation": obs, "info": info, "url": env.page.url})

        last_screenshots = [Image.fromarray(obs["image"])]
        obs_metadata = info['observation_metadata']

        root_node.trajectory = new_trajectory
        root_node._additional_info['observation_metadata'] = obs_metadata
        root_node._additional_info['last_screenshots'] = last_screenshots
        return

    def save_search_tree(self, root_state: Node, task_info: dict, result_dir: str):
        #### for later visualization
        logger.debug("Saving search tree visualization")
        task_id = task_info["task_id"]
        save_dir = os.path.join(result_dir, "search_trees", f"task_{task_id}")
        render_helper = MCTSRenderHelper(
            save_dir,
            os.path.join(result_dir, "tmp_images")
        )
        render_helper.render(root_state, self.Q)

        ### maybe for later training
        logger.debug("Saving raw search tree data")
        removed_node_data = {
            root_state: {'env': root_state.env}
        }
        root_state.env = None
        all_a_s = root_state._get_all_child_actions()
        for action, next_state in all_a_s:
            # remove data that cannot be pickled
            # clean the trajectory as theres a lot of duplicates
            removed_node_data[next_state] = {
                'env': next_state.env,
                'trajectory': next_state.trajectory
            }
            if next_state._lazy_initialized or action["action_type"] in [ActionTypes.STOP, ActionTypes.NONE]:
                new_tracj = []
            else:
                # only save current observation to save space
                new_tracj = [next_state.trajectory[-1]]

            next_state.env = None
            next_state.trajectory = new_tracj
        # save
        curr_time = time.strftime("%Y%m%d-%H%M%S")
        save_file = os.path.join(result_dir, "search_trees", f"task_{task_id}", f"tree_{curr_time}.pkl.xz")
        with lzma.open(save_file, "wb") as fwrite:
            pickle.dump(root_state, fwrite)
        # restore
        root_state.env = removed_node_data[root_state]['env']
        for action, next_state in all_a_s:
            next_state.env = removed_node_data[next_state]['env']
            next_state.trajectory = removed_node_data[next_state]['trajectory']
        return

    @atime_it
    async def anext_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        additional_inputs: dict[str, Any],
    ) -> Action:
        task_info = additional_inputs["task_info"]
        config_file = task_info["config_file"]
        root_action_history = additional_inputs["action_history"]
        env = additional_inputs["env"]
        cmd_args: AgentArguments = additional_inputs["cmd_args"]
        eval_args = additional_inputs["eval_args"]
        early_stop_fn = additional_inputs["early_stop_fn"]
        self.early_stop_fn = early_stop_fn

        vf_budget = cmd_args.vf_budget
        time_budget = cmd_args.time_budget
        if time_budget > 0:
            logger.info(f"Using time budget={time_budget} min")
            budget_tracker = SoftBudgetTracker(time_budget)
        else:
            logger.info(f"Using value function budget={vf_budget}")
            budget_tracker = SoftBudgetTracker(vf_budget)

        ### update search param
        self.value_function_model = cmd_args.value_function
        self.branching_factor = cmd_args.branching_factor

        ### initialize search
        self.reset(config_file)

        safe_trajectory = copy.deepcopy(trajectory)
        state_info: StateInfo = safe_trajectory[-1]  # type: ignore[assignment]
        root_state = Node(
            env=env,
            trajectory=safe_trajectory,
            action_trajectory=[],
            action_trajectory_str=[],
            action_set_tag=self.action_set_tag,
            value=0.0,
            children={},
            _additional_info={
                'intent': intent,
                'meta_data': copy.deepcopy(meta_data),
                'images': task_info["images"],
                'last_screenshots': [Image.fromarray(state_info["observation"]["image"])],
                'observation_metadata': copy.deepcopy(state_info['info']['observation_metadata'])
            },
            _lazy_initialized=False,
            _need_evaluation=False,
            is_root=True,
            is_terminal=False
        )
        final_action = create_none_action()
        await self.search(root_state)  # initialize the root node

        iteration_idx = 0
        start_time = time.time()
        while budget_tracker.get_remaining() > 0:
            logger.debug(f"MCTS Iteration {iteration_idx} with {budget_tracker.get_remaining():.2f} remaining budget")
            if iteration_idx != 0:
                # reset the env to root
                # the last reset can be skipped because the runner script will do the reset
                await self.reset_env(root_state, config_file, root_action_history)
                self.best_action_cache = []
            
            try:
                await self.search(root_state)
            except Exception as e:
                logger.error(e, exc_info=True)
                logger.error(f"Error in MCTS iteration {iteration_idx}")

            # early terminate if success trajectory is found
            if self.found_success_trajectory:
                logger.debug(f"Found success trajectory in {iteration_idx} iterations")
                final_action.metadata["all_candidates"] = root_state.get_all_children_str(self.Q)
                final_action.metadata["best_actions"] = self.best_action_cache
                final_action.metadata["best_score"] = 1.0
                self.save_search_tree(root_state, task_info, eval_args.result_dir)
                return final_action
            
            iteration_idx += 1

            ## update budget
            if time_budget > 0:
                budget_spent = (time.time() - start_time) / 60.0
                start_time = time.time()
            else:
                budget_spent = 1
            budget_tracker.spend(budget_spent)
        
        #### return best action from root node
        # alike AlphaGoZero, use the robust child policy = return the most visits
        best_action = create_none_action()
        best_value = -float('inf')
        best_visits = -1
        for a in root_state.children.keys():
            root_hashed = root_state._to_string_rep()
            if self.Nsa[root_hashed][a] > best_visits:
                best_visits = self.Nsa[root_hashed][a]
                best_value = self.Q[root_hashed][a]
                best_action = a
            elif self.Nsa[root_hashed][a] == best_visits:
                # tie break with Q value
                if self.Q[root_hashed][a] > best_value:
                    best_value = self.Q[root_hashed][a]
                    best_action = a
        final_action.metadata["all_candidates"] = root_state.get_all_children_str(self.Q)
        final_action.metadata["best_actions"] = [best_action]
        final_action.metadata["best_score"] = best_value

        #### save root node for later visualization
        self.save_search_tree(root_state, task_info, eval_args.result_dir)
        return final_action


    def reset(self, test_config_file: str) -> None:
        self.Ns = {}
        self.Nsa = {}
        self.Q = {}
        self.P = {}
        self.found_success_trajectory = False
        self.best_action_cache = []
        return
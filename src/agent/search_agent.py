import copy
import collections
import heapq
import time
import logging
from src.logging import atime_it, time_it
from typing import Any, Optional
from PIL import Image

from agent.prompts import PromptConstructor
from src.agentic.policy import MCoTPolicyPConstructor_OLD
from browser_env import Trajectory, ActionParsingError, ActionTypes
from src.envs.actions import (
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_stop_action
)
from src.helper_functions import get_action_description
from src.agentic import value_function
from src.envs.browser import early_stop, FastBrowserEnv, FastCachedBrowserEnv
from src.envs.actions import Action
from src.llms import lm_config
from src.llms.utils import call_llm, is_vlm
from src.agentic.value_function import ValueFunction
from src.agent.base_agent import FastAgent, Agent
from src.agent.agent_args import AgentArguments
from src.agent.utils import SoftBudgetTracker


logger = logging.getLogger('logger')


def maybe_update_best_action(curr_score, curr_actions, best_score, best_actions):
    # The second if statement checks that for scores < 1 that are tied, we should take the one that doesn't terminate.
    if (best_score is None) or (curr_score > best_score):
        best_score, best_actions = curr_score, curr_actions
    return best_score, best_actions


class SearchAgentRefactored(FastAgent):
    """refactored version of SearchAgent to improve readability"""
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        value_function: ValueFunction,
        prompt_constructor: PromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.value_function = value_function
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn

        # Check if the model is multimodal.
        if is_vlm(self.lm_config) and prompt_constructor.is_multimodal:
            self.multimodal_inputs = True
            logger.info("Using multimodal input in prompt.")
        else:
            self.multimodal_inputs = False
            logger.info("Model is not multimodal.")
        return

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag
        return

    @atime_it
    async def take_action_and_score(
        self,
        search_args,
        next_action: Action,
        curr_info: dict[str, Any],
        render_info: dict[str, Any],
        env: FastCachedBrowserEnv,
        task_info: dict[str, Any],
        meta_data,
        early_stop_fn,
        value_func_method: str,
        generate_next_actions: bool = True,
    ):
        """Take the given actions and score the resulting trajectory."""
        intent = task_info["intent"]
        task_id = task_info["task_id"]
        max_depth = search_args.max_depth
        branching_factor = search_args.branching_factor
        should_generate_next_actions = generate_next_actions

        temp_trajectory = copy.deepcopy(curr_info['curr_trajectory'])
        temp_action_history = copy.deepcopy(curr_info['curr_action_history'])
        curr_obs_metadata = copy.deepcopy(curr_info['curr_obs_metadata'])
        
        step_idx = render_info['step_idx']
        depth = render_info['curr_depth']
        a_idx = render_info['a_idx']
        curr_a_idx = render_info['curr_a_idx']
        img_after_path = f"{task_id}_step{step_idx}_action{a_idx}_depth{depth}_curra{curr_a_idx}_after.png"
        
        obs, _, terminated, _, info = await env.astep(next_action)

        # Save the image after the action.
        all_inputs = curr_info['all_inputs']
        obs_img = Image.fromarray(obs["image"])
        all_inputs.append({
            "text": obs["text"],
            "image": obs["image"],
            "image_after_path": img_after_path,
            "action": next_action["raw_prediction"],
        })
        temp_trajectory.append(next_action)

        # Get natural language description of current action
        curr_action_str = get_action_description(
            next_action, curr_obs_metadata,
            action_set_tag=self.action_set_tag,
            prompt_constructor=None,
        )
        temp_action_history.append(curr_action_str)

        if next_action["action_type"] == ActionTypes.STOP:
            should_generate_next_actions = False
        elif next_action["action_type"] != ActionTypes.STOP:
            temp_trajectory.append({"observation": obs, "info": info, "url": env.page.url})
        elif terminated:
            should_generate_next_actions = False
            temp_trajectory.append(create_stop_action(""))
        
        start_time = time.time()
        images = task_info["images"]
        # Only evaluate terminating trajectories
        try:
            logger.info(f"Evaluating current state at {depth=}")
            # last_screenshots = screenshots SINCE the root state
            last_screenshots = curr_info['last_screenshots'] + [obs_img]
            init_screenshot = last_screenshots[0]
            value_function_model = value_func_method  # search agent uses this param for model name
            if value_function_model in ["gpt4o"]:
                score = self.value_function.evaluate_success(
                    screenshots=last_screenshots,
                    actions=temp_action_history,
                    current_url=env.page.url,
                    last_reasoning="",
                    intent=intent,
                    models=["gpt-4o-2024-05-13"],
                    init_screenshot=init_screenshot,
                    intent_images=images if len(images) > 0 else None
                )
            elif value_function_model in ["gpt-4o", "gpt-4o-mini", "OpenGVLab/InternVL2-Llama3-76B"]:
                score = self.value_function.evaluate_success(
                    screenshots=last_screenshots,
                    actions=temp_action_history,
                    current_url=env.page.url,
                    last_reasoning="",
                    intent=intent,
                    models=[value_function_model],
                    init_screenshot=init_screenshot,
                    intent_images=images if len(images) > 0 else None
                )
            else:
                raise NotImplementedError(f"Value function {value_func_method} not implemented")
        except Exception as e:
            print(f"Error in evaluator: {e}")
            score = 0

        next_actions = []
        if score < 1 and should_generate_next_actions:
            temp_early_stop_flag, _ = early_stop_fn(temp_trajectory)
            if not temp_early_stop_flag:
                try:
                    # Generate possible action candidates for next step.
                    logger.info(f"Generating next actions at {depth=}")
                    logger.debug(f"Given intent: {intent} at {env.page.url}")
                    meta_data_copy = copy.deepcopy(meta_data)
                    meta_data_copy["action_history"] = temp_action_history
                    next_actions = self._gen_next_actions(
                        temp_trajectory,
                        intent,
                        images=images,
                        meta_data=meta_data_copy,
                        branching_factor=branching_factor
                    )
                except ValueError as e:
                    # get the error message
                    print('Failed to generate next actions:', e)

        return score, temp_trajectory, temp_action_history, next_actions

    @time_it
    def _gen_next_actions(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
        output_response: bool = False,
        branching_factor: int = 5
    ) -> list[Action]:
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        observation_metadata = state_info['info']['observation_metadata']

        if output_response:
            print("Using SearchAgent, branching_factor =", branching_factor)
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
                num_outputs=max(branching_factor * 2, 20)
            )
            if output_response:
                print(f'Agent: {responses}', flush=True)
            if type(responses) == str:
                responses = [responses]
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            n += 1
            all_actions = {}
            parsed_actions_count = {}

            for response in responses:
                response = f"{force_prefix}{response}"
                try:
                    parsed_response = self.prompt_constructor.extract_action(
                        response
                    )
                    if parsed_response in all_actions:
                        parsed_actions_count[parsed_response] += 1
                    else:
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
                        action['metadata']['obs_metadata'] = copy.deepcopy(observation_metadata)

                        parsed_actions_count[parsed_response] = 1
                        action["raw_prediction"] = response
                        all_actions[parsed_response] = action
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
                    return [action]
                
        # Find top branching_factor actions.
        top_actions = sorted(
            parsed_actions_count,
            key=parsed_actions_count.get, reverse=True
        )[:branching_factor]
        top_action_count = sum([parsed_actions_count[action] for action in top_actions])
        updated_actions = []
        for action in top_actions:
            a = all_actions[action]
            a['prob'] = parsed_actions_count[action] / top_action_count
            updated_actions.append(a)

        ### check if element id is found on the page
        logger.debug(f"Last turn prompt:\n")
        if isinstance(prompt[-1]['content'], list):
            # multiple messages
            for msg in prompt[-1]['content']:
                if 'text' in msg:
                    logger.debug(f" text: {msg['text']}")
                elif 'image_url' in msg:
                    logger.debug(f" image_url: {msg['image_url']['url'][:50]}...(truncated)")
        else:
            logger.debug(f" {prompt[-1]['content']}")
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

    @atime_it
    async def _anext_action(
        self,
        policy_next_actions,
        meta_data,
        trajectory,
        additional_inputs: dict[str, Any],
    ) -> Action:
        task_info = additional_inputs["task_info"]
        action_history: list = additional_inputs["action_history"]
        step_idx: int = additional_inputs["step_idx"]
        env = additional_inputs["env"]
        early_stop_fn = additional_inputs["early_stop_fn"]
        search_args: AgentArguments = additional_inputs["cmd_args"]

        ## search parameters
        max_depth = search_args.max_depth
        vf_budget = search_args.vf_budget
        time_budget = search_args.time_budget
        search_algo = search_args.search_algo
        value_func_model = search_args.value_function

        if time_budget > 0:
            logger.info(f"Using time budget={time_budget} min")
            budget_tracker = SoftBudgetTracker(time_budget)
        else:
            logger.info(f"Using value function budget={vf_budget}")
            budget_tracker = SoftBudgetTracker(vf_budget)

        ## code start here
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs = state_info["observation"]
        info = state_info["info"]

        # BEGIN SEARCH
        config_file = task_info["config_file"]
        
        best_score, best_actions = None, []
        all_candidates = []
        all_inputs = []  # For input to the value function computation.

        action_queue = []  # Store tuple of (score, a_idx, action, trajectory, depth, ...)
        actions_at_depth = collections.defaultdict(int)
        search_counter = 0
        for a_idx, a in enumerate(policy_next_actions):
            # Initialize the search queue. Default the score of the initial actions to 0.5.
            item = (
                -0.5, a_idx, -1,
                search_counter,
                [a],  # branching of actions from the root state
                copy.deepcopy(trajectory),  # real root state
                copy.deepcopy(meta_data["action_history"]),  # actions to get to the root state
                0
            )
            if search_algo == "bfs":
                action_queue.append(item)
            elif search_algo == "dfs":
                action_queue.append(item)
            elif search_algo == "vf":
                heapq.heappush(action_queue, item)
        
        ### MAIN SEARCH LOOP
        start_time = time.time()
        while action_queue and budget_tracker.get_remaining() > 0:
            logger.info(f"Search counter ({search_algo}): {search_counter}, remaining budget: {budget_tracker.get_remaining()}")

            if search_algo == "bfs":
                item = action_queue.pop(0)
            elif search_algo == "dfs":
                item = action_queue.pop(-1)
            elif search_algo == "vf":
                item = heapq.heappop(action_queue)
            (
                curr_score, a_idx, curr_a_idx,
                _,
                curr_actions,
                curr_trajectory,
                curr_action_history,
                curr_depth
            ) = item

            search_counter += 1
            next_action = curr_actions[-1]
            actions_at_depth[curr_depth] += 1
            assert len(curr_actions) == curr_depth + 1, f"(depth+1) should be equal to the number of actions taken, but got {len(curr_actions)} and {curr_depth+1}"

            if next_action["action_type"] == ActionTypes.NONE:
                best_score, best_actions = maybe_update_best_action(
                    0, curr_actions, best_score, best_actions
                )
            elif next_action["action_type"] == ActionTypes.STOP:
                best_score, best_actions = maybe_update_best_action(
                    curr_score, curr_actions, best_score, best_actions
                )
            else:
                last_screenshots = []
                # Reset environment to prepare for next action.
                _ = await env.areset(options={"config_file": config_file})
                # Take all the previous actions to get back to the current state.
                for a_hist in action_history:
                    obs, _, _, _, info = await env.astep(a_hist)
                last_screenshots.append(Image.fromarray(obs["image"]))

                # Take all previous actions in the current trajectory.
                start_time = time.time()
                for a in curr_actions[:-1]:
                    obs, _, _, _, info = await env.astep(a)
                    last_screenshots.append(Image.fromarray(obs["image"]))  # 1e-5s
                
                # Take the next action to evaluate.
                start_time = time.time()

                _curr_info = {
                    'curr_trajectory': curr_trajectory,
                    'curr_action_history': curr_action_history,
                    'curr_obs_metadata': info["observation_metadata"],
                    # 'last_screenshots': last_screenshots[-4:],  # TODO: this SHOULD be unnecessary
                    'last_screenshots': last_screenshots,
                    'all_inputs': all_inputs,
                }
                _render_info = {
                    'a_idx': a_idx,
                    'curr_a_idx': curr_a_idx,
                    'curr_depth': curr_depth,
                    'step_idx': step_idx,
                }
                score, new_trajectory, new_action_history, next_actions = await self.take_action_and_score(
                    search_args,
                    next_action,
                    curr_info=_curr_info,
                    render_info=_render_info,
                    env=env,
                    task_info=task_info,
                    meta_data=meta_data,
                    early_stop_fn=early_stop_fn,
                    value_func_method=value_func_model,
                    generate_next_actions=(curr_depth < max_depth),
                )

                raw_pred = next_action['raw_prediction'].split('\n')[-1]
                all_candidates.append(f'a_idx={a_idx},curr_a_idx={curr_a_idx},depth={curr_depth}: {next_action["raw_prediction"]} (score: {score}, time: {time.time() - start_time})')
                best_score, best_actions = maybe_update_best_action(
                    score, curr_actions,
                    best_score, best_actions
                )
                # Try for next action (if allowed)
                if score == 1:
                    break
                else:
                    # Add next actions to the queue.
                    for na_idx, na in enumerate(next_actions):
                        item = (
                            -score, a_idx, na_idx,
                            search_counter,
                            curr_actions + [na],
                            copy.deepcopy(new_trajectory),
                            copy.deepcopy(new_action_history),
                            curr_depth + 1
                        )
                        if search_algo == "vf":
                            heapq.heappush(action_queue, item)
                        else:
                            action_queue.append(item)
            # Update the budget tracker.
            if time_budget > 0:
                budget_spent = (time.time() - start_time) / 60.0
                start_time = time.time()
            else:
                budget_spent = 1
            budget_tracker.spend(budget_spent)

        none_action = create_none_action()
        none_action.metadata["all_candidates"] = all_candidates
        none_action.metadata["best_actions"] = best_actions
        none_action.metadata["best_score"] = best_score
        return none_action

    @atime_it
    async def anext_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        additional_inputs: dict[str, Any],
    ) -> Action:
        task_info = additional_inputs["task_info"]
        env = additional_inputs["env"]
        cmd_args = additional_inputs["cmd_args"]

        ## search parameters
        branching_factor = cmd_args.branching_factor
        output_response = False

        images = task_info["images"]

        logger.debug(f"Given intent: {intent} at {env.page.url}")
        next_actions = self._gen_next_actions(
            trajectory, intent, meta_data, images, output_response, branching_factor
        )

        final_actions = await self._anext_action(
            policy_next_actions=next_actions,
            meta_data=meta_data,
            trajectory=trajectory,
            additional_inputs=additional_inputs
        )
        return final_actions

    def reset(self, test_config_file: str) -> None:
        return



class SearchAgent(FastAgent):
    """searchagent from https://github.com/kohjingyu/search-agents but compatibile with async cached environment"""

    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn

        # Check if the model is multimodal.
        if ("gemini" in lm_config.model or "gpt-4" in lm_config.model and "vision" in lm_config.model or "gpt-4o" in lm_config.model) and type(prompt_constructor) == MCoTPolicyPConstructor_OLD:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False
        return

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag
        return

    @atime_it
    async def take_action_and_score(
        self,
        next_action: Action,
        curr_trajectory,
        curr_action_history,
        curr_obs_metadata,
        env,
        last_screenshots: list[Image.Image],
        task_id: int,
        step_idx: int,
        task_info,
        meta_data,
        value_func_method: str,
        max_depth: int,
        max_steps: int,
        early_stop_thresholds,
        all_inputs: list,
        generate_next_actions: bool = True,
        depth: int = 0,
        branching_factor: int = 5,
        a_idx: int = -1,
        curr_a_idx: int = -1
    ):
        """Take the given actions and score the resulting trajectory."""
        intent = task_info["intent"]

        temp_trajectory = copy.deepcopy(curr_trajectory)
        temp_action_history = copy.deepcopy(curr_action_history)
        should_generate_next_actions = generate_next_actions
        img_after_path = f"{task_id}_step{step_idx}_action{a_idx}_depth{depth}_curra{curr_a_idx}_after.png"
        
        obs, _, terminated, _, info = await env.astep(next_action)

        # Save the image after the action.
        obs_img = Image.fromarray(obs["image"])
        all_inputs.append({
            "text": obs["text"],
            "image": obs["image"],
            "image_after_path": img_after_path,
            "action": next_action["raw_prediction"],
        })
        temp_trajectory.append(next_action)

        # Get natural language description of current action
        curr_action_str = get_action_description(
            next_action, curr_obs_metadata,
            action_set_tag=self.action_set_tag,
            prompt_constructor=None,
        )
        temp_action_history.append(curr_action_str)

        if next_action["action_type"] == ActionTypes.STOP:
            should_generate_next_actions = False
        elif next_action["action_type"] != ActionTypes.STOP:
            temp_trajectory.append({"observation": obs, "info": info, "url": env.page.url})
        elif terminated:
            should_generate_next_actions = False
            temp_trajectory.append(create_stop_action(""))
        
        start_time = time.time()
        images = task_info["images"]
        # Only evaluate terminating trajectories
        try:
            if value_func_method in ["gpt4o"]:
                score = value_function.evaluate_success(
                    screenshots=last_screenshots[-(max_depth+1):] + [obs_img],
                    actions=temp_action_history,
                    current_url=env.page.url,
                    last_reasoning=next_action["raw_prediction"],
                    intent=intent,
                    models=["gpt-4o-2024-05-13"],
                    intent_images=images if len(images) > 0 else None
                )
            else:
                raise NotImplementedError(f"Value function {value_func_method} not implemented")
        except Exception as e:
            print(f"Error in evaluator: {e}")
            score = 0

        next_actions = []
        if score < 1 and should_generate_next_actions:
            temp_early_stop_flag, _ = early_stop(
                temp_trajectory, max_steps, early_stop_thresholds
            )
            if not temp_early_stop_flag:
                try:
                    # Generate possible action candidates for next step.
                    logger.debug(f"Given intent: {intent} at {env.page.url}")
                    next_actions = self._gen_next_actions(
                        temp_trajectory,
                        intent,
                        images=images,
                        meta_data=meta_data,
                        branching_factor=branching_factor
                    )
                except ValueError as e:
                    # get the error message
                    print('Failed to generate next actions:', e)

        return score, temp_trajectory, temp_action_history, next_actions

    @time_it
    def _gen_next_actions(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
        output_response: bool = False,
        branching_factor: int = 5
    ) -> list[Action]:
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        observation_metadata = state_info['info']['observation_metadata']

        if output_response:
            print("Using SearchAgent, branching_factor =", branching_factor)
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
                num_outputs=max(branching_factor * 2, 20)
            )
            if output_response:
                print(f'Agent: {responses}', flush=True)
            if type(responses) == str:
                responses = [responses]
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            n += 1
            all_actions = {}
            parsed_actions_count = {}

            for response in responses:
                response = f"{force_prefix}{response}"
                try:
                    parsed_response = self.prompt_constructor.extract_action(
                        response
                    )
                    if parsed_response in all_actions:
                        parsed_actions_count[parsed_response] += 1
                    else:
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
                        action['metadata']['obs_metadata'] = copy.deepcopy(observation_metadata)

                        parsed_actions_count[parsed_response] = 1
                        action["raw_prediction"] = response
                        all_actions[parsed_response] = action
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
                    return [action]
                
        # Find top branching_factor actions.
        top_actions = sorted(
            parsed_actions_count,
            key=parsed_actions_count.get, reverse=True
        )[:branching_factor]
        top_action_count = sum([parsed_actions_count[action] for action in top_actions])
        updated_actions = []
        for action in top_actions:
            a = all_actions[action]
            a['prob'] = parsed_actions_count[action] / top_action_count
            updated_actions.append(a)

        ### check if element id is found on the page
        logger.debug(f"Last turn prompt:\n{prompt[-1]['content']}")
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

    @atime_it
    async def _anext_action(
        self,
        policy_next_actions,
        task_info,
        meta_data,
        trajectory,
        action_history,
        step_idx: int,
        early_stop_thresholds,
        env,
        max_steps: int,
        branching_factor: int,
        max_depth: int,
        vf_budget: int,
        search_algo: str,
    ) -> Action:
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs = state_info["observation"]
        info = state_info["info"]

        # BEGIN SEARCH
        task_id = task_info["task_id"]
        config_file = task_info["config_file"]
        
        best_score, best_actions = None, []
        all_candidates = []
        all_inputs = []  # For input to the value function computation.

        action_queue = []  # Store tuple of (score, a_idx, action, trajectory, depth, ...)
        actions_at_depth = collections.defaultdict(int)
        search_counter = 0
        for a_idx, a in enumerate(policy_next_actions):
            # Initialize the search queue. Default the score of the initial actions to 0.5.
            item = (
                -0.5, a_idx, -1,
                search_counter,
                [a],  # branching of actions from the root state
                copy.deepcopy(trajectory),  # real root state
                copy.deepcopy(meta_data["action_history"]),  # actions to get to the root state
                0
            )
            if search_algo == "bfs":
                action_queue.append(item)
            elif search_algo == "dfs":
                action_queue.append(item)
            elif search_algo == "vf":
                heapq.heappush(action_queue, item)
        
        ### MAIN SEARCH LOOP
        while action_queue and search_counter < vf_budget:

            if search_algo == "bfs":
                item = action_queue.pop(0)
            elif search_algo == "dfs":
                item = action_queue.pop(-1)
            elif search_algo == "vf":
                item = heapq.heappop(action_queue)
            (
                curr_score, a_idx, curr_a_idx,
                _,
                curr_actions,
                curr_trajectory,
                curr_action_history,
                curr_depth
            ) = item

            search_counter += 1
            next_action = curr_actions[-1]
            actions_at_depth[curr_depth] += 1
            assert len(curr_actions) == curr_depth + 1, f"(depth+1) should be equal to the number of actions taken, but got {len(curr_actions)} and {curr_depth+1}"

            if next_action["action_type"] == ActionTypes.NONE:
                best_score, best_actions = maybe_update_best_action(
                    0, curr_actions, best_score, best_actions
                )
            elif next_action["action_type"] == ActionTypes.STOP:
                best_score, best_actions = maybe_update_best_action(
                    curr_score, curr_actions, best_score, best_actions
                )
            else:
                last_screenshots = []
                # Reset environment to prepare for next action.
                _ = await env.areset(options={"config_file": config_file})
                # Take all the previous actions to get back to the current state.
                for a_hist in action_history:
                    obs, _, _, _, info = await env.astep(a_hist)
                last_screenshots.append(Image.fromarray(obs["image"]))

                # Take all previous actions in the current trajectory.
                start_time = time.time()
                for a in curr_actions[:-1]:
                    obs, _, _, _, info = await env.astep(a)
                    last_screenshots.append(Image.fromarray(obs["image"]))  # 1e-5s
                
                # Take the next action to evaluate.
                start_time = time.time()
                score, new_trajectory, new_action_history, next_actions = await self.take_action_and_score(
                    next_action,
                    curr_trajectory,
                    curr_action_history,
                    copy.deepcopy(info["observation_metadata"]),
                    env=env,
                    last_screenshots=last_screenshots[-4:],
                    task_id=task_id,
                    step_idx=step_idx,
                    task_info=task_info,
                    meta_data=meta_data,
                    value_func_method="gpt4o",
                    max_depth=max_depth,
                    max_steps=max_steps,
                    early_stop_thresholds=early_stop_thresholds,
                    all_inputs=all_inputs,
                    generate_next_actions=(curr_depth < max_depth),
                    depth=curr_depth,
                    branching_factor=branching_factor,
                    a_idx=a_idx,
                    curr_a_idx=curr_a_idx
                )

                raw_pred = next_action['raw_prediction'].split('\n')[-1]
                all_candidates.append(f'a_idx={a_idx},curr_a_idx={curr_a_idx},depth={curr_depth}: {next_action["raw_prediction"]} (score: {score}, time: {time.time() - start_time})')
                best_score, best_actions = maybe_update_best_action(
                    score, curr_actions,
                    best_score, best_actions
                )
                # Try for next action (if allowed)
                if score == 1:
                    break
                else:
                    # Add next actions to the queue.
                    for na_idx, na in enumerate(next_actions):
                        item = (
                            -score, a_idx, na_idx,
                            search_counter,
                            curr_actions + [na],
                            copy.deepcopy(new_trajectory),
                            copy.deepcopy(new_action_history),
                            curr_depth + 1
                        )
                        if search_algo == "vf":
                            heapq.heappush(action_queue, item)
                        else:
                            action_queue.append(item)
        none_action = create_none_action()
        none_action.metadata["all_candidates"] = all_candidates
        none_action.metadata["best_actions"] = best_actions
        none_action.metadata["best_score"] = best_score
        return none_action

    @atime_it
    async def anext_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        ### additional arguments
        task_info: dict[str, Any],
        action_history: list[str],
        step_idx: int,
        early_stop_thresholds: dict[str, Any],
        env: FastCachedBrowserEnv,
        output_response: bool = False,
        max_steps: int = 30,
        max_depth: int = 4,
        branching_factor: int = 5,
        vf_budget: int = 20,
        search_algo: str = "vf",
    ) -> Action:
        images = task_info["images"]

        logger.debug(f"Given intent: {intent} at {env.page.url}")
        next_actions = self._gen_next_actions(
            trajectory, intent, meta_data, images, output_response, branching_factor
        )

        final_actions = await self._anext_action(
            policy_next_actions=next_actions,
            task_info=task_info,
            meta_data=meta_data,
            trajectory=trajectory,
            action_history=action_history,
            step_idx=step_idx,
            early_stop_thresholds=early_stop_thresholds,
            env=env,
            max_steps=max_steps,
            branching_factor=branching_factor,
            max_depth=max_depth,
            vf_budget=vf_budget,
            search_algo=search_algo
        )
        return final_actions

    def reset(self, test_config_file: str) -> None:
        return


#### for ablation studies
# SearchAgent + FastBrowserEnv
# to be run with runners/eval/eval_vwa_searchagent_v2.py
class AsyncSearchAgent(FastAgent):
    """prompt-based agent with search that emits action given the history"""

    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn

        # Check if the model is multimodal.
        if ("gemini" in lm_config.model or "gpt-4" in lm_config.model and "vision" in lm_config.model or "gpt-4o" in lm_config.model) and type(prompt_constructor) == MCoTPolicyPConstructor_OLD:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False
        return

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag
        return

    @atime_it
    async def take_action_and_score(
        self,
        a,
        curr_trajectory,
        curr_action_history,
        curr_obs_metadata,
        env,
        last_screenshots: list[Image.Image],
        task_id: int,
        step_idx: int,
        task_info,
        meta_data,
        value_func_method: str,
        max_depth: int,
        max_steps: int,
        early_stop_thresholds,
        all_inputs: list,
        generate_next_actions: bool = True,
        depth: int = 0,
        branching_factor: int = 5,
        a_idx: int = -1,
        curr_a_idx: int = -1
    ):
        """Take the given actions and score the resulting trajectory."""
        from browser_env.actions import (
            create_stop_action
        )

        intent = task_info["intent"]

        temp_trajectory = copy.deepcopy(curr_trajectory)
        temp_action_history = copy.deepcopy(curr_action_history)
        should_generate_next_actions = generate_next_actions
        img_after_path = f"{task_id}_step{step_idx}_action{a_idx}_depth{depth}_curra{curr_a_idx}_after.png"
        obs, _, terminated, _, info = await env.astep(a)
        # Save the image after the action.
        obs_img = Image.fromarray(obs["image"])
        all_inputs.append({
            "text": obs["text"],
            "image": obs["image"],
            "image_after_path": img_after_path,
            "action": a["raw_prediction"],
        })
        temp_trajectory.append(a)

        # Get natural language description of current action
        curr_action_str = get_action_description(
            a, curr_obs_metadata,
            action_set_tag=self.action_set_tag,
            prompt_constructor=None,
        )
        temp_action_history.append(curr_action_str)

        if a["action_type"] == ActionTypes.STOP:
            should_generate_next_actions = False
        elif a["action_type"] != ActionTypes.STOP:
            temp_trajectory.append({"observation": obs, "info": info, "url": env.page.url})
        elif terminated:
            should_generate_next_actions = False
            temp_trajectory.append(create_stop_action(""))
        
        start_time = time.time()
        images = task_info["images"]
        # Only evaluate terminating trajectories
        try:
            if value_func_method in ["gpt4o"]:
                score = value_function.evaluate_success(
                    screenshots=last_screenshots[-(max_depth+1):] + [obs_img],
                    actions=temp_action_history,
                    current_url=env.page.url,
                    last_reasoning=a["raw_prediction"],
                    intent=intent,
                    models=["gpt-4o-2024-05-13"],
                    intent_images=images if len(images) > 0 else None
                )
            else:
                raise NotImplementedError(f"Value function {value_func_method} not implemented")
        except Exception as e:
            print(f"Error in evaluator: {e}")
            score = 0

        next_actions = []
        if score < 1 and should_generate_next_actions:
            temp_early_stop_flag, _ = early_stop(
                temp_trajectory, max_steps, early_stop_thresholds
            )
            if not temp_early_stop_flag:
                try:
                    # Generate possible action candidates for next step.
                    logger.debug(f"Given intent: {intent} at {env.page.url}")
                    next_actions = self._gen_next_actions(
                        temp_trajectory,
                        intent,
                        images=images,
                        meta_data=meta_data,
                        branching_factor=branching_factor
                    )
                except ValueError as e:
                    # get the error message
                    print('Failed to generate next actions:', e)

        return score, temp_trajectory, temp_action_history, next_actions

    @time_it
    def _gen_next_actions(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
        output_response: bool = False,
        branching_factor: int = 5
    ) -> list[Action]:
        from browser_env.actions import (
            ActionParsingError,
            create_id_based_action,
            create_none_action,
            create_playwright_action,
        )
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs_nodes_info = state_info['info']['observation_metadata']['text']['obs_nodes_info']

        if output_response:
            print("Using SearchAgent, branching_factor =", branching_factor)
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
                print(
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
                num_outputs=max(branching_factor * 2, 20)
            )
            if output_response:
                print(f'Agent: {responses}', flush=True)
            if type(responses) == str:
                responses = [responses]
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            n += 1
            all_actions = {}
            parsed_actions_count = {}

            for response in responses:
                response = f"{force_prefix}{response}"
                try:
                    parsed_response = self.prompt_constructor.extract_action(
                        response
                    )
                    if parsed_response in all_actions:
                        parsed_actions_count[parsed_response] += 1
                    else:
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
                        # TODO: maybe add text info to action for recovery later
                        if action.get('metadata', None) is None:
                            action['metadata'] = {}
                        action['metadata']['obs_nodes_info_keys'] = copy.deepcopy(list(obs_nodes_info.keys()))

                        parsed_actions_count[parsed_response] = 1
                        action["raw_prediction"] = response
                        all_actions[parsed_response] = action
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
                    return [action]
                
        # Find top branching_factor actions.
        top_actions = sorted(
            parsed_actions_count,
            key=parsed_actions_count.get, reverse=True
        )[:branching_factor]
        top_action_count = sum([parsed_actions_count[action] for action in top_actions])
        updated_actions = []
        for action in top_actions:
            a = all_actions[action]
            a['prob'] = parsed_actions_count[action] / top_action_count
            updated_actions.append(a)

        ### check if element id is found on the page
        logger.debug(f"Last turn prompt:\n{prompt[-1]['content']}")
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

    @atime_it
    async def _anext_action(
        self,
        policy_next_actions,
        task_info,
        meta_data,
        trajectory,
        action_history,
        step_idx: int,
        early_stop_thresholds,
        env,
        max_steps: int,
        branching_factor: int,
        max_depth: int,
        vf_budget: int,
        search_algo: str,
    ) -> Action:
        from browser_env import create_none_action

        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs = state_info["observation"]
        info = state_info["info"]

        # BEGIN SEARCH
        task_id = task_info["task_id"]
        config_file = task_info["config_file"]
        
        best_score, best_actions = None, []
        all_candidates = []
        all_inputs = []  # For input to the value function computation.

        action_queue = []  # Store tuple of (score, a_idx, action, trajectory, depth, ...)
        actions_at_depth = collections.defaultdict(int)
        search_counter = 0
        for a_idx, a in enumerate(policy_next_actions):
            # Initialize the search queue. Default the score of the initial actions to 0.5.
            item = (
                -0.5, a_idx, -1,
                search_counter,
                [a],
                copy.deepcopy(trajectory),
                copy.deepcopy(meta_data["action_history"]),
                0
            )
            if search_algo == "bfs":
                action_queue.append(item)
            elif search_algo == "dfs":
                action_queue.append(item)
            elif search_algo == "vf":
                heapq.heappush(action_queue, item)
        
        ### MAIN SEARCH LOOP
        while action_queue and search_counter < vf_budget:

            if search_algo == "bfs":
                item = action_queue.pop(0)
            elif search_algo == "dfs":
                item = action_queue.pop(-1)
            elif search_algo == "vf":
                item = heapq.heappop(action_queue)
            (
                curr_score, a_idx, curr_a_idx,
                _,
                curr_actions,
                curr_trajectory,
                curr_action_history,
                curr_depth
            ) = item

            search_counter += 1
            next_action = curr_actions[-1]
            actions_at_depth[curr_depth] += 1
            assert len(curr_actions) == curr_depth + 1, f"(depth+1) should be equal to the number of actions taken, but got {len(curr_actions)} and {curr_depth+1}"

            if next_action["action_type"] == ActionTypes.NONE:
                best_score, best_actions = maybe_update_best_action(
                    0, curr_actions, best_score, best_actions
                )
            elif next_action["action_type"] == ActionTypes.STOP:
                best_score, best_actions = maybe_update_best_action(
                    curr_score, curr_actions, best_score, best_actions
                )
            else:
                last_screenshots = []
                # Reset environment to prepare for next action.
                _ = await env.areset(options={"config_file": config_file})
                # Take all the previous actions to get back to the current state.
                for a_hist in action_history:
                    obs, _, _, _, info = await env.astep(a_hist)
                last_screenshots.append(Image.fromarray(obs["image"]))

                # Take all previous actions in the current trajectory.
                start_time = time.time()
                for a in curr_actions[:-1]:
                    obs, _, _, _, info = await env.astep(a)
                    last_screenshots.append(Image.fromarray(obs["image"]))  # 1e-5s
                
                # Take the next action to evaluate.
                start_time = time.time()
                score, new_trajectory, new_action_history, next_actions = await self.take_action_and_score(
                    next_action,
                    curr_trajectory,
                    curr_action_history,
                    copy.deepcopy(info["observation_metadata"]),
                    env=env,
                    last_screenshots=last_screenshots[-4:],
                    task_id=task_id,
                    step_idx=step_idx,
                    task_info=task_info,
                    meta_data=meta_data,
                    value_func_method="gpt4o",
                    max_depth=max_depth,
                    max_steps=max_steps,
                    early_stop_thresholds=early_stop_thresholds,
                    all_inputs=all_inputs,
                    generate_next_actions=(curr_depth < max_depth),
                    depth=curr_depth,
                    branching_factor=branching_factor,
                    a_idx=a_idx,
                    curr_a_idx=curr_a_idx
                )

                raw_pred = next_action['raw_prediction'].split('\n')[-1]
                all_candidates.append(f'a_idx={a_idx},curr_a_idx={curr_a_idx},depth={curr_depth}: {next_action["raw_prediction"]} (score: {score}, time: {time.time() - start_time})')
                best_score, best_actions = maybe_update_best_action(
                    score, curr_actions,
                    best_score, best_actions
                )
                # Try for next action (if allowed)
                if score == 1:
                    break
                else:
                    # Add next actions to the queue.
                    for na_idx, na in enumerate(next_actions):
                        item = (
                            -score, a_idx, na_idx,
                            search_counter,
                            curr_actions + [na],
                            copy.deepcopy(new_trajectory),
                            copy.deepcopy(new_action_history),
                            curr_depth + 1
                        )
                        if search_algo == "vf":
                            heapq.heappush(action_queue, item)
                        else:
                            action_queue.append(item)
        none_action = create_none_action()
        none_action['metadata'] = {
            "all_candidates": all_candidates,
            "best_actions": best_actions,
            "best_score": best_score
        }
        return none_action

    @atime_it
    async def anext_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        ### additional arguments
        task_info: dict[str, Any],
        action_history: list[str],
        step_idx: int,
        early_stop_thresholds: dict[str, Any],
        env: FastBrowserEnv,
        output_response: bool = False,
        max_steps: int = 30,
        max_depth: int = 4,
        branching_factor: int = 5,
        vf_budget: int = 20,
        search_algo: str = "vf",
    ) -> Action:
        images = task_info["images"]

        logger.debug(f"Given intent: {intent} at {env.page.url}")
        next_actions = self._gen_next_actions(
            trajectory, intent, meta_data, images, output_response, branching_factor
        )

        final_actions = await self._anext_action(
            policy_next_actions=next_actions,
            task_info=task_info,
            meta_data=meta_data,
            trajectory=trajectory,
            action_history=action_history,
            step_idx=step_idx,
            early_stop_thresholds=early_stop_thresholds,
            env=env,
            max_steps=max_steps,
            branching_factor=branching_factor,
            max_depth=max_depth,
            vf_budget=vf_budget,
            search_algo=search_algo
        )
        return final_actions

    def reset(self, test_config_file: str) -> None:
        return



# the basic SearchAgent from prior work
# to be run with runners/eval/eval_vwa_searchagent.py
class BaseSearchAgent(Agent):
    """prompt-based agent with search that emits action given the history"""

    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn

        # Check if the model is multimodal.
        if ("gemini" in lm_config.model or "gpt-4" in lm_config.model and "vision" in lm_config.model or "gpt-4o" in lm_config.model) and type(prompt_constructor) == MCoTPolicyPConstructor_OLD:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @time_it
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any], images: Optional[list[Image.Image]] = None,
        output_response: bool = False, branching_factor: int = 5
    ) -> list[Action]:
        from browser_env.actions import (
            ActionParsingError,
            create_id_based_action,
            create_none_action,
            create_playwright_action,
        )
        if output_response:
            print("Using SearchAgent, branching_factor =", branching_factor)
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
                print(
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
            responses = call_llm(lm_config, prompt, num_outputs=max(branching_factor * 2, 20))
            if output_response:
                print(f'Agent: {responses}', flush=True)
            if type(responses) == str:
                responses = [responses]
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            n += 1
            all_actions = {}
            parsed_actions_count = {}

            for response in responses:
                response = f"{force_prefix}{response}"
                try:
                    parsed_response = self.prompt_constructor.extract_action(
                        response
                    )
                    if parsed_response in all_actions:
                        parsed_actions_count[parsed_response] += 1
                    else:
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
                        parsed_actions_count[parsed_response] = 1
                        action["raw_prediction"] = response
                        all_actions[parsed_response] = action
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
                    return [action]
                
        # Find top branching_factor actions.
        top_actions = sorted(parsed_actions_count, key=parsed_actions_count.get, reverse=True)[:branching_factor]
        top_action_count = sum([parsed_actions_count[action] for action in top_actions])
        updated_actions = []
        for action in top_actions:
            a = all_actions[action]
            a['prob'] = parsed_actions_count[action] / top_action_count
            updated_actions.append(a)

        return updated_actions

    def reset(self, test_config_file: str) -> None:
        pass
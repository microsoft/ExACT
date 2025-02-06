import logging
import os
import datetime
import json
import concurrent.futures
from typing import Dict, List
from dataclasses import dataclass, field
from exact.llms.lm_config import LMConfig
from exact.llms.tokenizer import Tokenizer
from exact.llms.utils import (
    configure_llm_client, call_llm, 
    _truncate_prompt_to_max_tokens, _force_truncate_prompt_to_max_tokens,
    _flatten_chat_msg_turns
)
from exact.logging import time_it
from exact.env.desktop_env_dev import PooledDesktopEnv
from exact.env.desktop_env_utils import (
    ObsPostProcessor,
    parse_actions_from_string, parse_code_from_string, parse_code_from_som_string
)
from exact.prompts.utils import configure_system_prompt
from exact.agentic.policy_prompt import ReACTPolicyArgs, ReACTPolicy
from exact.agent.base import BaseAgent
from exact.args import AgentArgs, EnvArgs



logger = logging.getLogger("src.agent")


@dataclass
class RejectionSamplingAgentArgs(AgentArgs):
    agent: str = "rjn_sampling"
    n_trajectory: int = field(
        default=5, metadata={"help": "Number of trajectories to sample"}
    )
    max_steps_per_trajectory: int = field(
        default=15, metadata={"help": "Maximum number of steps per trajectory"}
    )

    def __post_init__(self):
        super().__post_init__()
        return


@dataclass
class RejectionSamplingSearchMetadata:
    ## env
    env: PooledDesktopEnv
    env_args: EnvArgs
    task_config: dict

    ## from common args
    result_dir: str


class RejectionSamplingAgent(BaseAgent):
    name: str = "rjn_sampling"

    def __init__(
        self,
        args: RejectionSamplingAgentArgs,
        action_space="computer_13",
        observation_type="screenshot_a11y_tree",
        platform="ubuntu"
    ):
        self.args = args
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type

        self.thoughts = []
        self.actions = []
        self.observations = []

        system_message = configure_system_prompt(observation_type, action_space, self.args.user_prompt_prefix)
        self.system_message = system_message
        
        self.lm_config, self.llm_client = self._configure_client()
        self._obs_processor = ObsPostProcessor(
            observation_type=observation_type,
            platform=self.platform,
            a11y_tree_max_tokens=self.args.a11y_tree_max_tokens
        )
        # policy
        ### this logic should be refactored to be inside agent_factory
        if self.args.policy == ReACTPolicyArgs.name:
            self.policy_prompt = ReACTPolicy(
                args=ReACTPolicyArgs(
                    max_trajectory_length=self.args.max_trajectory_length,
                    user_prompt_prefix=self.args.user_prompt_prefix,
                ),
                system_message=self.system_message,
                observation_type=self.observation_type,
                action_space=self.action_space,
            )
        else:
            raise ValueError("Invalid policy: " + self.args.policy)
        return

    @property
    def obs_processor(self):
        return self._obs_processor

    def _configure_client(self):
        model_name = self.args.model
        lm_config = LMConfig(
            provider=self.args.model_api_provider,
            model=model_name,
            mode="chat",
            tokenizer_cls=Tokenizer(
                provider=self.args.model_api_provider,
                model_name=model_name,
                max_context_length=self.args.max_context_length,
            ),
            api_base=os.environ.get("POLICY_LLM_API_BASE", "http://127.0.0.1:30000/v1"),
            api_key=os.environ.get("POLICY_LLM_API_KEY", "empty"),
            api_version=os.environ.get("POLICY_LLM_API_VERSION", ""),
            api_token_provider_base=os.environ.get("POLICY_LLM_TOKEN_PROVIDER_BASE", ""),
            gen_config={
                'temperature': self.args.temperature,
                'top_p': self.args.top_p,
                'max_tokens': self.args.max_tokens,
            }
        )
        client = configure_llm_client(lm_config)
        return lm_config, client

    @time_it
    def _gen_next_action(self, instruction: str, obs: Dict) -> List:
        """
        Predict the next action(s) based on the current observation.
        """
        processed_obs = self.obs_processor(obs)
        self.observations.append(processed_obs)
        messages = self.policy_prompt.get_messages(
            instruction=instruction,
            past_obs=self.observations,
            past_actions=self.actions,
            past_thoughts=self.thoughts,
        )

        masks = None
        try:
            response = self.call_llm(messages)
        except Exception as e:
            logger.error("Failed to call" + self.args.model + ", Error: " + str(e))
            response = ""

        logger.info("RESPONSE: %s", response)

        try:
            actions = self.parse_actions(response, masks)
            self.thoughts.append(response)
            self.actions.append(actions)
        except ValueError as e:
            print("Failed to parse action from response", e)
            actions = None
            self.thoughts.append("")
            self.actions.append(actions)
        return response, actions

    def _get_env_idx_for_simulation(self, env: PooledDesktopEnv, task_config=None) -> int:
        # for rejection sampling, this is simply the next available env
        free_env_idx = env._get_unused_env_ids(task_config=task_config)
        return free_env_idx[0]

    def search(
        self,
        search_idx: int,
        obs: Dict,
        env: PooledDesktopEnv,
        env_args: EnvArgs,
        result_dir: str,
        instruction: str,
        task_config: dict,
        max_steps: int = 15,
    ):
        traj_responses = []
        traj_actions = []

        ## init search metadata
        search_save_dir = os.path.join(result_dir, f"search_{search_idx}")
        os.makedirs(search_save_dir, exist_ok=True)
        step_idx = 0
        value = 0.0
        simu_env_idx = self._get_env_idx_for_simulation(env, task_config=task_config)
        done = False
        while not done and step_idx < max_steps:
            response, actions = self._gen_next_action(
                instruction,
                obs
            )
            traj_responses.append(response)
            traj_actions.append(actions)

            for action in actions:
                # Capture the timestamp before executing the action
                action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                logger.info("Step %d: %s", step_idx + 1, action)
                obs, reward, done, info = env.simu_step(
                    action,
                    env_idx=simu_env_idx,
                    pause=env_args.sleep_after_execution
                )

                logger.info(f"Reward: {reward:.2f} at {search_idx=}")
                logger.info(f"Done: {done} at {search_idx=}")
                
                # Save screenshot and trajectory information
                screenshot_fpath = os.path.join(search_save_dir, f"step_{step_idx + 1}_{action_timestamp}.png")
                with open(screenshot_fpath,"wb") as _fwrite:
                    _fwrite.write(obs['screenshot'])
                with open(os.path.join(search_save_dir, "traj.jsonl"), "a") as f:
                    f.write(json.dumps({
                        "step_num": step_idx + 1,
                        "action_timestamp": action_timestamp,
                        "action": action,
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                    }))
                    f.write("\n")
                if done:
                    logger.info(f"The episode is done at {search_idx=}")
                    break
            step_idx += 1
        
        value = env.simu_evaluate(env_idx=simu_env_idx)
        value_fpath = os.path.join(search_save_dir, "value.json")
        with open(value_fpath, "w") as fwrite:
            json.dump({
                "value": float(value),
                "actual_score": float(value),
            }, fwrite)

        ## calling render_trajectory_to_html includes a lot of refactoring. But we can just save the responses
        all_resp_fpath = os.path.join(search_save_dir, "all_responses.txt")
        with open(all_resp_fpath, "w") as fwrite:
            fwrite.write("\n\nNEXT RESP:\n".join(traj_responses))
        
        return traj_responses, traj_actions, value

    @time_it
    def predict(self, instruction: str, obs: Dict, search_metadata: RejectionSamplingSearchMetadata) -> List:
        """rejection sampling
        Return the action trajectories that reached the goal, if found
        else, return the last action trajectory
        """
        # simple version, do it sequentially

        n_trajectory = self.args.n_trajectory
        best_value = float("-inf")
        best_actions = []
        best_responses = []

        ## check if there is enough env for simuation
        free_env_idx = search_metadata.env._get_unused_env_ids(task_config=search_metadata.task_config)
        assert len(free_env_idx) >= n_trajectory, f"Not enough env for simulation. Got {free_env_idx=}, need {n_trajectory=} envs."

        # multithreading doesn't work since we also uses signal for timeouts
        # and signal only works in the main thread
        # multiprocessing also doesn't work since we cannot pickle certain objects
        for search_idx in range(n_trajectory):
            logger.info(f"Search {search_idx=}")
            self.reset()  # clear the thoughts, actions, and observations

            responses, actions, value = self.search(
                search_idx=search_idx,
                obs=obs,
                env=search_metadata.env,
                env_args=search_metadata.env_args,
                result_dir=search_metadata.result_dir,
                instruction=instruction,
                task_config=search_metadata.task_config,
                max_steps=self.args.max_steps_per_trajectory,
            )
            if value > best_value:
                best_value = value
                best_actions = actions
                best_responses = responses

                logger.info(f"Found a better trajectory with value {value}")
                logger.info(f"Trajectory: {actions}")
            if value == 1.0:
                logger.info(f"Found a trajectory with value 1.0. Early stopping search.")
                break
        return best_responses, best_actions

    @time_it
    def call_llm(self, messages):
        ## 1. truncate the prompt instead of simple left truncate if we let tokenizer do it
        if self.args.force_context_truncation:
            messages = _force_truncate_prompt_to_max_tokens(
                prompt=messages,
                tokenizer=self.lm_config.tokenizer_cls,
            )
        else:
            messages = _truncate_prompt_to_max_tokens(
                prompt=messages,
                tokenizer=self.lm_config.tokenizer_cls
            )
        ## 2. flatten the chat messages (useful if the models are also TRAINED this way)
        if self.args.flatten_chat_msg:
            engine = self.args.flatten_engine
            messages = _flatten_chat_msg_turns(messages, engine=engine)
        
        ## 3. call the model
        response = call_llm(self.llm_client, self.lm_config, messages)
        return response

    def parse_actions(self, response: str, masks=None):
        # NOTE: I moved self.actions.append(actions) to the predict function
        if self.observation_type in ["screenshot", "a11y_tree", "screenshot_a11y_tree"]:
            # parse from the response
            if self.action_space == "computer_13":
                actions = parse_actions_from_string(response)
            elif self.action_space == "pyautogui":
                actions = parse_code_from_string(response)
            else:
                raise ValueError("Invalid action space: " + self.action_space)
            return actions
        elif self.observation_type in ["som"]:
            # parse from the response
            if self.action_space == "computer_13":
                raise ValueError("Invalid action space: " + self.action_space)
            elif self.action_space == "pyautogui":
                actions = parse_code_from_som_string(response, masks)
            else:
                raise ValueError("Invalid action space: " + self.action_space)
            return actions
        else:
            raise ValueError("Invalid observation_type type: " + self.observation_type)

    def reset(self):
        self.thoughts = []
        self.actions = []
        self.observations = []
        return

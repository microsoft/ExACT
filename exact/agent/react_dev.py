import logging
import os
from typing import Dict, List
from exact.llms.lm_config import LMConfig
from exact.llms.tokenizer import Tokenizer
from exact.llms.utils import (
    configure_llm_client, call_llm, 
    _truncate_prompt_to_max_tokens, _force_truncate_prompt_to_max_tokens,
    _flatten_chat_msg_turns
)
from exact.env.desktop_env_utils import (
    ObsPostProcessor,
    parse_actions_from_string, parse_code_from_string, parse_code_from_som_string
)
from exact.prompts.utils import configure_system_prompt
from exact.agentic.policy_prompt import ReACTPolicyArgs, ReACTPolicy
from exact.agent.base import BaseAgent
from exact.logging import time_it
from exact.args import AgentArgs


logger = logging.getLogger("src.agent")


class PromptAgent(BaseAgent):
    name: str = "react"

    def __init__(
        self,
        args: AgentArgs,
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
    def predict(self, instruction: str, obs: Dict, search_metadata=None) -> List:
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
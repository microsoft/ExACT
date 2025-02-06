import logging
import os
import jsonlines
import lzma
import gzip
import pickle
import copy
import numpy as np
import math
import time
import textwrap
import ast
import threading
import datetime
from scipy.special import softmax
from graphviz import Digraph
from PIL import Image
from io import BytesIO
from collections import defaultdict
from typing import Dict, List, Any
from dataclasses import dataclass, field
from exact.llms.lm_config import LMConfig
from exact.llms.tokenizer import Tokenizer
from exact.llms.utils import (
    configure_llm_client, call_llm, 
    _truncate_prompt_to_max_tokens, _force_truncate_prompt_to_max_tokens,
    _flatten_chat_msg_turns
)
from exact.llms.providers.openai_utils import get_all_token_usage, set_all_token_usage
from exact.logging import time_it
from exact.env.utils import timeout
from exact.env.desktop_env_dev import PooledDesktopEnv, DynamicPooledDesktopEnv
from exact.env.desktop_env_utils import (
    ObsPostProcessor,
    parse_actions_from_string, parse_code_from_string, parse_code_from_som_string
)
from exact.prompts.utils import configure_system_prompt
from exact.agentic.policy_prompt import ReACTPolicyArgs, ReACTPolicy
from exact.agentic.value_function import ValueFunction
from exact.agent.base import BaseAgent, ResumableAgentMixin
from exact.args import AgentArgs, EnvArgs



logger = logging.getLogger("src.agent")


@dataclass
class MCTSAgentArgs(AgentArgs):
    agent: str = "mcts"
    cpuct: float = field(
        default=1.0, metadata={"help": "UCT exploration constant"}
    )
    n_nodes: int = field(
        default=5, metadata={"help": "Number of trajectories to sample"}
    )
    branching_factor: int = field(
        default=5, metadata={"help": "Maximum number of children per node"}
    )
    branching_algo: str = field(
        default="best", metadata={
            "choices": ["best", "sample", "random"],
            "help": "Algorithm to select branching factor"
        }
    )
    bfactor_func: str = field(
        default="constant", metadata={
            "choices": ["constant", "exp_decay"],
            "help": "Function to use for branching factor"
        }
    )
    bfactor_func_coeff: float = field(
        default=1.0, metadata={"help": "Coefficient for branching factor function. Has different meaning for diff funcs."}
    )
    prior_temperature: float = field(
        default=1.0, metadata={"help": "Temperature to convert frequency to prior probability"}
    )
    adv_after_n_nodes: int = field(
        default=10_000, metadata={"help": "Advance root node after its Ns reaches this value"}
    )
    adv_counter: str = field(
        default="search_itr",
        metadata={
            "choices": ["search_itr", "subtree_size"],
            "help": "Counter to use for advancing root node"
        }
    )
    c_func: str = field(
        default="constant", metadata={
            "choices": ["constant", "linear", "exp_decay", "cosine"],
            "help": "Function to use for exploration"
        }
    )
    cpuct_end: float = field(
        default=1.0, metadata={"help": "end value for cpuct when using exp_decay or cosine"}
    )

    def __post_init__(self):
        super().__post_init__()
        return


@dataclass
class MCTSAgentSearchMetadata:
    ## env
    env: DynamicPooledDesktopEnv
    env_args: EnvArgs
    task_config: dict   # used for creating new simluation envs

    ## from common args
    result_dir: str



@dataclass
class Node:
    env: DynamicPooledDesktopEnv
    env_args: EnvArgs

    response_trajectory: list[str]
    observations: list[dict]  # all obs from start until now
    past_responses: list[str]  # since we are evaluating observation, all responses before this observation
    past_actions: list[dict]
    value: float
    children: dict[str, 'Node']   # dict[response, next_node]. note that a response can have multiple actions
    parent: 'Node'
    Ps: dict[str, float]  # prior probability of the children responses
    ## helper info
    _resp_to_action: dict[str, list[str]] = field(default_factory=dict)  # helper dict to map response to parsed actions
    _additional_info: dict[str, Any] = field(default_factory=dict)
    _need_simluation_n_eval: bool = True
    _lazy_expanded: bool = True  # to save compute, we only generate children when next action is needed
    Ns: int = 0 # num visits
    depth: int = 0
    is_root: bool = False
    is_terminal: bool = False

    def _to_string_rep(self) -> str:
        # assumes that same action sequence will lead to the same state
        return '\n--->\n'.join(self.response_trajectory)

    def _get_all_children(self) -> list[tuple[str, 'Node']]:
        # recursively get all children
        child_state_actions = []
        for a, s in self.children.items():
            child_state_actions.append((a, s))
            child_actions = s._get_all_children()
            child_state_actions.extend(child_actions)
        return child_state_actions

    def _get_simulated_subtree_size(self) -> int:
        """returns the REAL number of subtree nodes that have been simulated
        this may be different from Ns, since root_node may move and after that, that root_node's Ns will NOT be updated

        Returns:
            int: _description_
        """
        all_a_s = self._get_all_children()
        num_simluated = 0
        for _, s in all_a_s:
            if not s._need_simluation_n_eval:
                num_simluated += 1
        return num_simluated

    def get_all_children_str(self, Q: dict = None) -> list[str]:
        all_children = []
        all_a_s = self._get_all_children()
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
    return textwrap.fill(long_str, 90, break_long_words=False, replace_whitespace=False)


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
        curr_screenshot = state._additional_info['curr_raw_obs']['screenshot']
        image_to_display = Image.open(BytesIO(curr_screenshot))
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

            _has_real_state = next_s._additional_info['curr_raw_obs'] is not None
            if not _has_real_state:
                # next state image not rendered
                curr_node_idx = len(seen_nodes) + 1
                seen_nodes[child_hash] = curr_node_idx
                graph.node(f'{curr_node_idx}', label=f"lazy init")
            elif child_hash not in seen_nodes:
                self._dfs_nodes(next_s, graph, Q, seen_nodes)
            
            # format label to display
            a_prob = state.Ps[a]
            actual_exec_a = state._resp_to_action[a]
            value_str = f"Q={qsa:.2f}, V next={next_s.value}, N next={next_s.Ns}, p(Action)={a_prob:.2f}"
            act_str = "Action=\l"
            for a_idx, actual_a in enumerate(actual_exec_a):
                actual_a = break_long_string(actual_a).strip().replace('\n', '\l')
                act_str += f"{a_idx}: {actual_a}\l"
            reason_str = break_long_string(f"Reasoning=\l{a}").strip().replace('\n', '\l')
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


class MCTSAgent(BaseAgent, ResumableAgentMixin):
    name: str = "mcts"

    def __init__(
        self,
        args: MCTSAgentArgs,
        value_function: ValueFunction,
        action_space="computer_13",
        observation_type="screenshot_a11y_tree",
        platform="ubuntu"
    ):
        ResumableAgentMixin.__init__(self)

        self.args = args
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type

        system_message = configure_system_prompt(observation_type, action_space, self.args.user_prompt_prefix)
        self.system_message = system_message
        
        self.lm_config, self.llm_client = self._configure_client()
        self._obs_processor = ObsPostProcessor(
            observation_type=observation_type,
            platform=self.platform,
            a11y_tree_max_tokens=self.args.a11y_tree_max_tokens
        )
        # policy
        self.policy_prompt = ReACTPolicy(
            args=ReACTPolicyArgs(
                max_trajectory_length=self.args.max_trajectory_length,
                user_prompt_prefix=self.args.user_prompt_prefix,
            ),
            system_message=self.system_message,
            observation_type=self.observation_type,
            action_space=self.action_space,
        )
        self.value_function = value_function

        ## mcts related quantity
        self.cpuct = self.args.cpuct
        self.branching_factor = self.args.branching_factor
        self.root_node = None
        self.Ns = {}
        self.Nsa = {}
        self.Q = {}
        self.P = {}
        self.found_success_trajectory = False
        self._success_response_cache = []
        self._success_action_cache = []
        self._traversal_cache = []
        self._action_hist_to_env_idx = {}

        ## resuming related
        self._search_itr_to_resume = 0
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
                max_context_length=self.args.max_context_length
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

    def _actions_to_unique_keys(self, parsed_action: list[str] | str):
        if self.action_space == "computer_13":
            if isinstance(parsed_action, str):
                return tuple([parsed_action])
            return tuple(parsed_action)
        elif self.action_space == "pyautogui":
            action_list = parsed_action
            if isinstance(parsed_action, str):
                action_list = [parsed_action]

            ast_parsed_actions = []
            for a in action_list:
                try:
                    parsed = ast.parse(a)  # remove comments
                    ast_parsed_actions.append(ast.dump(parsed))
                except Exception as e:
                    logger.warning(f"Failed to parse action in {a} using ast")
                    ast_parsed_actions.append(a)
            return tuple(ast_parsed_actions)
        else:
            raise ValueError("Invalid action space: " + self.action_space)

    def _get_unique_responses_and_actions(self, responses: list[str]):
        seen_response_count = {}
        seen_response_keys = {}
        resp_to_parsed_actions = {}
        masks = None
        for response in responses:
            try:
                action = self.parse_actions(response, masks)
            except ValueError as e:
                logger.error("Failed to parse action from response", e)
                action = []
            resp_to_parsed_actions[response] = action

            resp_key = self._actions_to_unique_keys(action)
            if resp_key not in seen_response_count:
                seen_response_count[resp_key] = 0
            seen_response_count[resp_key] += 1

            if resp_key not in seen_response_keys:
                seen_response_keys[resp_key] = response

        unique_resp = []
        corresponding_actions = []
        counts = []
        for resp_key, response in seen_response_keys.items():
            unique_resp.append(response)
            corresponding_actions.append(resp_to_parsed_actions[response])
            counts.append(seen_response_count[resp_key])
        return unique_resp, corresponding_actions, counts

    @time_it
    def _gen_next_action(
        self,
        instruction: str,
        past_obs: list,
        past_actions: list,
        past_thoughts: list
    ) -> tuple[list, list, list]:
        """
        Predict the next action(s) based on the current observation.
        """
        messages = self.policy_prompt.get_messages(
            instruction=instruction,
            past_obs=past_obs,
            past_actions=past_actions,
            past_thoughts=past_thoughts,
        )

        try:
            responses: list[str] = self.call_llm(messages)
        except Exception as e:
            logger.error("Failed to call" + self.args.model + ", Error: " + str(e))
            responses = []

        logger.debug("received RESPONSE")
        for r_idx, r in enumerate(responses):
            logger.debug(f"Response {r_idx}: {r}")

        unique_responses, actionses, counts = self._get_unique_responses_and_actions(responses)
        return unique_responses, actionses, counts

    def _get_env_idx_for_simulation(self, env: DynamicPooledDesktopEnv, state: Node) -> int:
        # find the next available env to go forward
        # - if a parent exist, go from there
        #   this works because one iteration of MCTS is basically finding expanding ONE node from an EXISTING parent in the tree
        # - if immediate parent does NOT exist, recurisvely go back until a parent is found
        #   then remember to replay the missing actions
        task_config = state._additional_info['task_config']
        
        response_trace = state.response_trajectory
        if len(response_trace) == 0:
            env_idx = env._get_unused_env_ids(task_config)[0]
        else:
            logger.debug(f"Finding env to play the last action in {response_trace}")
            parent_responses = response_trace[:-1]
            parent_node = state.parent
            env_key = tuple(parent_responses)

            logger.debug(f"Current cache: {self._action_hist_to_env_idx}")
            _responses_to_replay = []   # for debugging
            _nested_actions_to_replay = []  # NOT nested list
            
            while True:
                if len(parent_responses) == 0:
                    env_idx = env._get_unused_env_ids(task_config)[0]
                    logger.debug(f"Go back to empty parent. Starting anew with {env_idx=}")
                    break
                if env_key in self._action_hist_to_env_idx:
                    env_idx = self._action_hist_to_env_idx[env_key]
                    logger.debug(f"Parent response {parent_responses} found in cache. Continuing with {env_idx=}")
                    break
                logger.debug(f"Parent response {parent_responses} not found in cache. Backing off")
                ## replay this response
                _r_to_replay = parent_responses[-1]
                _responses_to_replay.append(_r_to_replay)
                parent_node = parent_node.parent
                a_to_replay = parent_node._resp_to_action[_r_to_replay]
                _nested_actions_to_replay.append(a_to_replay)

                ## and go back
                parent_responses = parent_responses[:-1]
                env_key = tuple(parent_responses)
            
            _responses_to_replay = _responses_to_replay[::-1]  # real sequence
            _nested_actions_to_replay = _nested_actions_to_replay[::-1]
            logger.debug(f"Replaying responses: {_responses_to_replay}")
            logger.debug(f"Replaying actions: {_nested_actions_to_replay}")

            for actions in _nested_actions_to_replay:
                for a in actions:
                    obs, reward, done, info = env.simu_step(
                        a,
                        env_idx=env_idx,
                        pause=state.env_args.sleep_after_execution
                    )
                    logger.debug(f'received {info=}, {reward=}, {done=}')
                    if done:
                        logger.info(f"The episode is done after replaying actions: {actions}")
                        break
        return env_idx

    def _set_env_idx_after_simulation(self, state: Node, env_idx: int):
        # set where we are
        response_trace = state.response_trajectory
        curr_env_key = tuple(response_trace)
        logger.debug(f"Setting {env_idx=} to {curr_env_key}")
        ## set it reversely, because the real shared thing is the env_idx
        _env_idx_to_action_hist = {v: k for k, v in self._action_hist_to_env_idx.items()}
        _env_idx_to_action_hist[env_idx] = curr_env_key
        self._action_hist_to_env_idx = {v: k for k, v in _env_idx_to_action_hist.items()}
        logger.debug(f"Cache after setting {self._action_hist_to_env_idx}")
        return

    def _simulation(self, state: Node):
        logger.info("Simulating state value")
        if state.is_root:
            return 0.0
        
        try:
            v = self.value_function.predict(
                instruction=state._additional_info['instruction'],
                obss=state.observations,
                actions=state.past_actions,
                thoughts=state.past_responses
            )
        except Exception as e:
            logger.error("Failed to parse value")
            logger.error(e, exc_info=True)
            v = 0.0
        
        state.value = v
        state._need_simluation_n_eval = False
        return v
    
    def _maybe_update_branching_factor(
        self,
        curr_depth: int
    ) -> int:
        if self.args.bfactor_func == "constant":
            return self.args.branching_factor
        elif self.args.bfactor_func == "exp_decay":
            coefficient = np.log(2) / self.args.bfactor_func_coeff
            branching_factor = self.args.branching_factor * np.exp(-coefficient * curr_depth)
            return max(2, int(branching_factor))
        else:
            raise ValueError("Invalid branching_factor_func: " + self.args.bfactor_func)

    def _expansion(self, state: Node):
        ## expansion
        logger.info("Expanding state")

        responses, actionses, counts = self._gen_next_action(
            instruction=state._additional_info['instruction'],
            past_actions=state.past_actions,
            past_obs=state.observations,
            past_thoughts=state.past_responses
        )
        
        branching_factor = self._maybe_update_branching_factor(state.depth)
        logger.debug(f"Branching factor: {branching_factor} at depth {state.depth}")
        if self.args.branching_algo == "best":
            top_k_response_indices = np.argsort(counts)[-branching_factor:]
        elif self.args.branching_algo == "sample":
            if len(responses) <= branching_factor:
                top_k_response_indices = list(range(len(responses)))
            else:
                dist = np.array(counts) / sum(counts)
                # faithful to LM distribution
                top_k_response_indices = np.random.choice(
                    len(responses),
                    size=branching_factor,
                    p=dist,
                    replace=False
                )
        elif self.args.branching_algo == "random":
            if len(responses) <= branching_factor:
                top_k_response_indices = list(range(len(responses)))
            else:
                top_k_response_indices = np.random.choice(
                    len(responses),
                    size=branching_factor,
                    replace=False
                )
        else:
            raise ValueError("Invalid branching_algo: " + self.args.branching_algo)

        top_k_responses = []
        top_k_actions = []
        top_k_counts = []
        for i in top_k_response_indices:
            top_k_responses.append(responses[i])
            top_k_actions.append(actionses[i])
            top_k_counts.append(counts[i])
        
        ## encourage UCT to explore
        prob = np.array(top_k_counts) / sum(top_k_counts)
        logger.debug(f"Top k responses freq distribution: {prob}")
        prob = softmax(np.log(prob) / self.args.prior_temperature)
        logger.debug(f"Top k responses prob distribution: {prob}")
        ps = {r: p for r, p in zip(top_k_responses, prob)}
        state.Ps = ps

        _resp_to_action = {}
        for r, a in zip(top_k_responses, top_k_actions):
            _resp_to_action[r] = a
        state._resp_to_action = _resp_to_action

        # create children
        for r in top_k_responses:
            new_node = Node(
                env=state.env,
                env_args=state.env_args,
                response_trajectory=state.response_trajectory + [r], # this is basically deepcopy
                observations=copy.deepcopy(state.observations), # note that observations is missing the new observation until _get_next_state
                past_responses=state.past_responses + [r],
                past_actions=state.past_actions + [_resp_to_action[r]],
                value=0.0,
                depth=state.depth + 1,
                children={},
                parent=state,
                Ps={},
                _resp_to_action={},
                _additional_info={
                    'instruction': state._additional_info['instruction'],
                    'curr_raw_obs': None,
                    'task_config': state._additional_info['task_config'],
                    'Q': {}
                },
                _need_simluation_n_eval=True,
                _lazy_expanded=True,
                is_root=False,
                is_terminal=False
            )
            state.children[r] = new_node
        #### now this node is expanded, and should never come here again
        state._lazy_expanded = False

        state_str_hash = state._to_string_rep()
        logger.debug(f"expanded state: {state_str_hash} with len={len(prob)} child actions")
        if len(prob) == 0:
            logger.error(f"expanded state: {state_str_hash} with no children. This will cause error later.")
        assert state_str_hash not in self.Ns, f"state {state_str_hash} already expanded"
        self.Ns[state_str_hash] = 0
        state.Ns = 0
        self.Nsa[state_str_hash] = defaultdict(int)  # instead of lambda: 0.0 which cannot be pickled
        self.Q[state_str_hash] = defaultdict(int)  # 0.0 for Q[s][new_a]
        self.P[state_str_hash] = ps
        return

    def _get_next_state(self, state: Node, response: str) -> Node:
        logger.debug(f"Entering _get_next_state with {state.response_trajectory=}")
        assert state._lazy_expanded is False, "state should not be lazy expanded"
        
        env = state.env
        env_args = state.env_args
        next_state = state.children[response]
        # the children need simulation if it does not have a raw observation,
        # because we will call value function on it next
        if next_state._additional_info['curr_raw_obs'] is None:
            actions = state._resp_to_action[response]

            simu_env_idx = self._get_env_idx_for_simulation(env, next_state)
            logger.debug(f'_get_next_state executing response: {response} with {len(actions)} actions')
            done = False
            obs = state._additional_info['curr_raw_obs']
            for action in actions:
                obs, reward, done, info = env.simu_step(
                    action,
                    env_idx=simu_env_idx,
                    pause=env_args.sleep_after_execution
                )
                logger.debug(f'received {info=}, {reward=}, {done=}')
                if done:
                    logger.info(f"The episode is done after executing response: {response}")
                    break

            # only the last obs is really needed
            self._set_env_idx_after_simulation(next_state, simu_env_idx)

            processed_obs = self.obs_processor(obs) # if this is errored, add_info is not updated
            next_state._additional_info['curr_raw_obs'] = obs
            next_state.observations.append(processed_obs)
            next_state.is_terminal = True if done else False
        return next_state

    def search(self, state: Node):
        """perform one iteration of MCTS: selection, expansion, simulation, backpropagation
        """
        state_str_hash = state._to_string_rep()
        
        v = 0.0
        # if this leaf node is terminal, return the value
        if state.is_terminal:
            # terminal node
            logger.debug(f"reached terminal state: {state_str_hash}")
            if state._need_simluation_n_eval:
                self._simulation(state)
            return state.value
        elif state.value == 1.0:
            best_resp_str = "\n--->\n".join(self._success_response_cache)
            logger.info(f"found task finishing trajectory: {best_resp_str}")
            _same_response_traj = state.past_responses
            logger.info(f"debug trajectory from the node: {_same_response_traj}")
            self.found_success_trajectory = True
            return state.value
        elif len(state.children) == 0:
            # selected leaf node, expand and simulate (for backprop below)
            self._expansion(state)  # Can we save token usage here by only expanding when we are going to visit its children?
            if state._need_simluation_n_eval:
                self._simulation(state)
            return state.value
        
        ##### Selection
        # existing, continue selection
        # go next state by picking best according to U(s,a)
        best_uct = -float('inf')
        best_response = None
        for a in state.children.keys():
            Ns = self.Ns[state_str_hash]
            qsa = self.Q[state_str_hash][a]
            p = self.P[state_str_hash][a]
            nsa = self.Nsa[state_str_hash][a]
            if Ns == 0:  # first time visit
                uct = qsa + self.cpuct * p
            else:
                uct = qsa + self.cpuct * p * math.sqrt(Ns) / (1 + nsa)
            
            if uct > best_uct:
                best_uct = uct
                best_response = a
                logger.debug(f"updating best action: {best_response}")
                logger.debug(f"uct={uct} (with {Ns=}, {nsa=}, {qsa=}, {p=})")
        logger.debug(f"selected best action: {best_response}")
        self._traversal_cache.append({
            'curr_state_str': state_str_hash,
            'best_response': best_response,
            'best_uct': best_uct,
            '_Ns': self.Ns[state_str_hash],
            '_qsa': self.Q[state_str_hash][best_response],
            '_p': self.P[state_str_hash][best_response],
            '_nsa': self.Nsa[state_str_hash][best_response]
        })
        
        # transition and update that state's metadata
        self._success_response_cache.append(best_response)
        self._success_action_cache.append(state._resp_to_action[best_response])
        logger.debug(f"current _success_response_cache: {self._success_response_cache}")
        next_state = self._get_next_state(state, best_response)
        
        ##### Expansion and Simulation
        # 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
        # 2. if leaf, we will expand it and return the value for backpropagation
        v = self.search(next_state)

        ##### Backpropagation
        # update stats
        # add in new estimate and average
        nsa = self.Nsa[state_str_hash][best_response]
        self.Q[state_str_hash][best_response] = (nsa * self.Q[state_str_hash][best_response] + v) / (nsa + 1)
        logger.debug(f"backpropagating value {v} to get Q[{state_str_hash}][{best_response}]={self.Q[state_str_hash][best_response]}")
        self.Nsa[state_str_hash][best_response] += 1
        self.Ns[state_str_hash] += 1
        state.Ns += 1
        state._additional_info['Q'] = {k: v for k, v in self.Q[state_str_hash].items()}

        if v == 1.0:
            best_resp_str = "\n--->\n".join(self._success_response_cache)
            logger.info(f"found task finishing trajectory: {best_resp_str}")
            _same_response_traj = next_state.past_responses
            logger.info(f"debug trajectory from the node: {_same_response_traj}")
            self.found_success_trajectory = True
        return v

    def _better_node_metric(self, node_a: Node, node_b: Node) -> bool:
        """true if node_a is better than node_b

        Args:
            node_a (Node): _description_
            node_b (Node): _description_

        Returns:
            bool: _description_
        """
        # prefer the one that has been simulated
        if node_a._need_simluation_n_eval and not node_b._need_simluation_n_eval:
            return False
        elif not node_a._need_simluation_n_eval and node_b._need_simluation_n_eval:
            return True
        
        # most robust child, with a value being tied breaker
        node_a_visit = node_a.Ns
        node_b_visit = node_b.Ns
        node_a_value = node_a.value
        node_b_value = node_b.value
        
        if node_a_visit > node_b_visit:
            return True
        elif node_a_visit == node_b_visit:
            if node_a_value > node_b_value:
                return True
        return False

    def _advance_root_state(self, node: Node) -> Node:
        """used to advance root node. Note that it calls _get_next_state in case its child was not simulated yet

        Args:
            node (Node): _description_

        Returns:
            Node: _description_
        """
        best_node = None
        best_response = None
        for a, s in node.children.items():
            if best_node is None or self._better_node_metric(s, best_node):
                best_node = s
                best_response = a
        
        if best_node is None:
            logger.info("No child found, returning current node")
            return node
        
        logger.debug("calling _get_next_state to advance root node")
        best_child = self._get_next_state(node, best_response)  # probably doesn't need this, but just in case
        return best_child

    def get_best_trajectory(self, root_node: Node) -> List:
        # strategy: recursively follow the highest "scored" node
        best_responses = []
        best_actions = []

        curr_node = root_node
        while len(curr_node.children) > 0:
            best_rep = None
            best_node = None
            for a, s in curr_node.children.items():
                # if visits > best_visits or (visits == best_visits and v > best_value):
                if best_node is None or self._better_node_metric(s, best_node):
                    best_rep = a
                    best_node = s
            
            best_responses.append(best_rep)
            best_actions.append(curr_node._resp_to_action[best_rep])
            curr_node = best_node
        return best_responses, best_actions

    def _maybe_update_cpuct(
        self,
        curr_search_idx: int
    ) -> float:
        if self.args.c_func == "constant":
            return self.args.cpuct
        elif self.args.c_func == "linear":
            coefficient = 1 - self.args.cpuct_end
            return 1 - coefficient * curr_search_idx / self.args.n_nodes
        elif self.args.c_func == "exp_decay":
            coefficient = np.log(self.args.cpuct_end) / self.args.n_nodes
            return np.exp(coefficient * curr_search_idx)
        elif self.args.c_func == "cosine":
            divider = self.args.n_nodes / (np.pi / 2)
            coefficient = np.arccos(self.args.cpuct_end) * (2 / np.pi)
            return np.cos(coefficient * curr_search_idx / divider)
        else:
            raise ValueError("Invalid c_func: " + self.args.c_func)

    def _maybe_advance_root_node(
        self,
        curr_search_idx: int,
        curr_root_node: Node
    ):
        if self.args.adv_counter == "search_itr":
            if (curr_search_idx + 1) % self.args.adv_after_n_nodes == 0:
                logger.info(f"Advancing root node after {curr_search_idx} iterations")
                curr_root_node = self._advance_root_state(curr_root_node)
                logger.info(f"advanced to root node response: {curr_root_node.response_trajectory}")
        elif self.args.adv_counter == "subtree_size":
            # subtree_size = curr_root_node._get_simulated_subtree_size()
            subtree_size = curr_root_node.Ns
            logger.debug(f"current node subtree size: {subtree_size}")
            if subtree_size >= self.args.adv_after_n_nodes:
                logger.info(f"Advancing root node after {curr_search_idx} iterations, subtree size {subtree_size}")
                curr_root_node = self._advance_root_state(curr_root_node)
                # new_subtree_size = curr_root_node._get_simulated_subtree_size()
                new_subtree_size = curr_root_node.Ns
                logger.info(f"advanced to root node response: {curr_root_node.response_trajectory}, new size {new_subtree_size}")
        else:
            raise ValueError("Invalid adv_counter: " + self.args.adv_counter)
        return curr_root_node

    @time_it
    def predict(self, instruction: str, obs: Dict, search_metadata: MCTSAgentSearchMetadata) -> List:
        """MCTS
        return the best trajectory according to most robust child
        """
        n_iterations = self.args.n_nodes

        ## check if there is enough env for simuation
        if isinstance(search_metadata.env, DynamicPooledDesktopEnv):
            n_sim_envs = search_metadata.env._actual_sim_instances
            logger.debug(f"DynamicPooledDesktopEnv detected with {n_sim_envs=}. _get_unused_env_ids will create new envs on the fly, if needed")
        elif isinstance(search_metadata.env, PooledDesktopEnv):
            free_env_idx = search_metadata.env._get_unused_env_ids()
            assert len(free_env_idx) >= n_iterations, f"Not enough env for simulation. Got {free_env_idx=}, need {self.args.n_nodes=} envs."

        ## initialize root state
        processed_obs = self.obs_processor(obs)
        root_state = Node(
            env=search_metadata.env,
            env_args=search_metadata.env_args,
            response_trajectory=[],
            observations=[processed_obs],
            past_responses=[],
            past_actions=[],
            value=0.0,
            children={},
            parent=None,
            Ps={},
            _resp_to_action={},
            _additional_info={
                'instruction': instruction,
                'curr_raw_obs': copy.deepcopy(obs),
                'task_config': search_metadata.task_config,
                'Q': {}
            },
            _need_simluation_n_eval=False,
            _lazy_expanded=True,
            is_root=True,
            is_terminal=False
        )
        self.root_node = root_state
        
        traversal_cache = []
        curr_root_state = root_state
        for search_idx in range(n_iterations):  # no +1 so that we can recover "ReACT" when n_iterations=1
            logger.info(f"Search {search_idx=}")
            self.cpuct = self._maybe_update_cpuct(search_idx)
            logger.debug(f"current cpuct: {self.cpuct}")

            # record traversal order for later exploratory learning
            self._traversal_cache = []
            self._success_response_cache = copy.deepcopy(curr_root_state.past_responses)
            self._success_action_cache = copy.deepcopy(curr_root_state.past_actions)

            ##### MCTS
            # note that this will modify _traversal_cache, _success_response_cache, _success_action_cache
            self.search(curr_root_state)
            self._search_itr_to_resume = search_idx + 1
            
            # post search
            traversal_cache.append(self._traversal_cache)

            if self.found_success_trajectory:
                logger.debug(f"Found success trajectory in {search_idx} iterations")
                self.save_search_tree(root_state, traversal_cache, search_metadata.result_dir)
                logger.debug(f"Returning success responses: {self._success_response_cache}")
                return self._success_response_cache, self._success_action_cache

            ### advance root node per n_nodes
            curr_root_state = self._maybe_advance_root_node(search_idx, curr_root_state)
        
        best_responses, best_actions = self.get_best_trajectory(self.root_node)
        logger.debug(f"Returning best responses: {best_responses}")
        self.save_search_tree(self.root_node, traversal_cache, search_metadata.result_dir)
        return best_responses, best_actions

    @time_it
    def resume_predict(self, instruction: str, obs: Dict, search_metadata: MCTSAgentSearchMetadata) -> List:
        n_iterations = self.args.n_nodes
        logger.info(f"Resuming search with {self._search_itr_to_resume=} completed")

        ## check if there is enough env for simuation
        if isinstance(search_metadata.env, DynamicPooledDesktopEnv):
            n_sim_envs = search_metadata.env._actual_sim_instances
            logger.debug(f"DynamicPooledDesktopEnv detected with {n_sim_envs=}. _get_unused_env_ids will create new envs on the fly, if needed")
        elif isinstance(search_metadata.env, PooledDesktopEnv):
            free_env_idx = search_metadata.env._get_unused_env_ids()
            assert len(free_env_idx) >= n_iterations, f"Not enough env for simulation. Got {free_env_idx=}, need {self.args.n_nodes=} envs."

        ## initialize root state
        assert self.root_node is not None, "Root node is not initialized"
        root_state = self.root_node
        root_state.env = search_metadata.env
        root_state.env_args = search_metadata.env_args
        # restore the env var
        all_a_s = root_state._get_all_children()
        for _, next_state in all_a_s:
            next_state.env = search_metadata.env
            next_state.env_args = search_metadata.env_args
        
        traversal_cache = []
        curr_root_state = root_state

        ## recover curr_root_state, which may have moved
        for search_idx in range(self._search_itr_to_resume):
            ### advance root node per n_nodes
            logger.info(f"Fast forwarding root node after {search_idx} iterations")
            curr_root_state = self._maybe_advance_root_node(search_idx, curr_root_state)
        
        new_start_idx = self._search_itr_to_resume
        for search_idx in range(new_start_idx, n_iterations):  # no +1 so that we can recover "ReACT" when n_iterations=1
            logger.info(f"Search {search_idx=}")
            self.cpuct = self._maybe_update_cpuct(search_idx)
            logger.debug(f"current cpuct: {self.cpuct}")

            # record traversal order for later exploratory learning
            self._traversal_cache = []
            self._success_response_cache = copy.deepcopy(curr_root_state.past_responses)
            self._success_action_cache = copy.deepcopy(curr_root_state.past_actions)

            ##### MCTS
            # note that this will modify _traversal_cache, _success_response_cache, _success_action_cache
            self.search(curr_root_state)
            self._search_itr_to_resume = search_idx + 1
            
            # post search
            traversal_cache.append(self._traversal_cache)

            if self.found_success_trajectory:
                logger.debug(f"Found success trajectory in {search_idx} iterations")
                self.save_search_tree(root_state, traversal_cache, search_metadata.result_dir)
                logger.debug(f"Returning success responses: {self._success_response_cache}")
                return self._success_response_cache, self._success_action_cache

            ### advance root node per n_nodes
            curr_root_state = self._maybe_advance_root_node(search_idx, curr_root_state)
        
        best_responses, best_actions = self.get_best_trajectory(self.root_node)
        logger.debug(f"Returning best responses: {best_responses}")
        self.save_search_tree(self.root_node, traversal_cache, search_metadata.result_dir)
        return best_responses, best_actions
    
    def _save_sate(self, save_path, data_dict):
        with gzip.open(save_path, "wb", compresslevel=6) as fwrite:
            pickle.dump(data_dict, fwrite)
        return
    
    @time_it
    def save_state(self, save_path: str):
        # remove data that cannot be pickled
        # these will be restored inside resume_predict
        if self.root_node is None:
            logger.warning("Root node is not initialized. Probably crashed before any search")
            return
        
        self.root_node.env = None
        all_a_s = self.root_node._get_all_children()
        for _, next_state in all_a_s:
            next_state.env = None
        
        time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        all_token_usages_so_far = get_all_token_usage() # otherwise resuming will screw up the token calculation
        important_states = {
            'name': self.name,
            'root_node': self.root_node,
            'Ns': self.Ns,
            'Nsa': self.Nsa,
            'Q': self.Q,
            'P': self.P,
            '_search_itr_to_resume': self._search_itr_to_resume,
            '_token_usage': {
                time_now: all_token_usages_so_far
            },
        }
        logger.debug(f"Saving agent state to {save_path}")
        ### do not hang the program if its taking too long to save
        thread = threading.Thread(
            target=self._save_sate,
            args=(save_path, important_states),
            daemon=True
        )
        thread.start()
        thread.join(timeout=60*10)
        if thread.is_alive():
            logger.info("Hoping self._save_sate can finish its job later. Continuing program.")
        return

    @time_it
    def load_state(self, save_path: str) -> None:
        logger.debug(f"Loading agent state from {save_path}")
        # can throw error if the file is corrupted
        with gzip.open(save_path, "rb", compresslevel=6) as fread:
            important_states = pickle.load(fread)
        
        self.root_node = important_states['root_node']
        self.Ns = important_states['Ns']
        self.Nsa = important_states['Nsa']
        self.Q = important_states['Q']
        self.P = important_states['P']
        self._search_itr_to_resume = important_states['_search_itr_to_resume']

        ### restore token usage
        past_token_usage = important_states.get('_token_usage', {})
        summed_token_usage = {}
        for _, token_usage in past_token_usage.items():
            for m_name, m_stats in token_usage.items():
                if m_name not in summed_token_usage:
                    summed_token_usage[m_name] = m_stats
                else:
                    new_completion_ts = m_stats['completion_tokens']
                    new_total_ts = m_stats['prompt_tokens']
                    new_n_requests = m_stats['num_requests']

                    summed_token_usage[m_name]['completion_tokens'] += new_completion_ts
                    summed_token_usage[m_name]['prompt_tokens'] += new_total_ts
                    summed_token_usage[m_name]['num_requests'] += new_n_requests
        set_all_token_usage(summed_token_usage)
        return

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
        response = call_llm(
            self.llm_client,
            self.lm_config,
            messages,
            num_outputs=max(self.args.branching_factor * 2, 20)
        )
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

    def _save_tree_from_root(self, root_state: Node, result_dir: str):
        removed_node_data = {
            root_state: {'env': root_state.env}
        }
        root_state.env = None
        all_a_s = root_state._get_all_children()
        for action, next_state in all_a_s:
            # remove data that cannot be pickled
            # clean the trajectory as theres a lot of duplicates
            next_state: Node
            removed_node_data[next_state] = {
                'env': next_state.env,
                'observations': next_state.observations,
            }
        
            # only save current observation to save space
            new_observations = [next_state.observations[-1]]

            next_state.env = None
            next_state.observations = new_observations
        # save
        save_file = os.path.join(result_dir, f"search_tree.pkl.xz")
        with lzma.open(save_file, "wb") as fwrite:
            pickle.dump(root_state, fwrite)
        
        # restore
        root_state.env = removed_node_data[root_state]['env']
        for action, next_state in all_a_s:
            next_state.env = removed_node_data[next_state]['env']
            next_state.observations = removed_node_data[next_state]['observations']
        return

    def save_search_tree(self, root_state: Node, traversal_cache: list, result_dir: str):
        #### for later visualization
        logger.info("Saving search tree visualization")
        save_dir = os.path.join(result_dir, "search_viz")
        render_helper = MCTSRenderHelper(
            save_dir,
            tmp_image_save_dir=save_dir
        )
        render_helper.render(root_state, self.Q)

        ### saving search sequences
        logger.debug("Saving search sequences")
        save_file = os.path.join(result_dir, f"search_sequences.jsonl")
        with jsonlines.open(save_file, "w") as fwrite:
            fwrite.write_all(traversal_cache)

        ### maybe for later training
        logger.debug("Saving raw search tree data")
        self._save_tree_from_root(root_state, result_dir)
        return

    def reset(self):
        self.root_node = None
        self.Ns = {}
        self.Nsa = {}
        self.Q = {}
        self.P = {}
        self.found_success_trajectory = False
        self._success_response_cache = []
        self._success_action_cache = []
        self._traversal_cache = []
        self._action_hist_to_env_idx = {}
        
        # resume related
        self._search_itr_to_resume = 0
        return
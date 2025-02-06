from typing import Any
from graphviz import Digraph
from exact.agent.base import RAgentMixin
from exact.agent.mcts import (
    MCTSAgent, MCTSAgentArgs, MCTSRenderHelper, Node,
    break_long_string
)
from exact.agentic.types import TaskRecord
from exact.agentic.value_function import ValueFunction
from exact.agentic.rpolicy_prompt import (
    ReinforcedReACTPolicy, ReinforcedReACTPolicyArgs,
    ReinforcedPolicyMixin, get_rpolicy_retrieval_key
)
from exact.agentic.rvalue_function import ReinforcedValueFunctionMixin
from exact.logging import time_it
from dataclasses import dataclass, field
import lzma
import pickle
import numpy as np
import os
import logging
import jsonlines


logger = logging.getLogger("src.agent")


@dataclass
class RMCTSAgentArgs(MCTSAgentArgs):
    agent: str = "rmcts"

    ### model config
    embedding_model: str = field(
        default="text-embedding-3-small",
        metadata={"help": "The model to use for embedding past experience."}
    )
    embedding_api_provider: str = field(
        default="openai",
        metadata={"help": "The API provider for the embedding model."}
    )
    rlm_model: str = field(
        default="gpt-4o",
        metadata={"help": "The model to use for generating reflections."}
    )
    rlm_api_provider: str = field(
        default="openai",
        metadata={"help": "The API provider for the reflection model."}
    )
    rlm_temperature: float = field(
        default=0.7,
        metadata={"help": "The temperature to use for the reflection model."}
    )
    rlm_top_p: float = field(
        default=0.9,
        metadata={"help": "The top-p value to use for the reflection model."}
    )
    rlm_max_tokens: int = field(
        default=400,
        metadata={"help": "The maximum number of tokens to generate for the reflection model."}
    )
    rlm_force_context_truncation: bool = False
    rlm_flatten_chat_msg: bool = False
    rlm_flatten_engine: str = "vllm"
    rlm_max_context_length: int = 0

    ### reflection/retrieval related
    db_path: str = field(
        default=None,
        metadata={"help": "Path to the directory where the task records (and reflection dbs) are stored. If none, store under result root dir."}
    )
    selection_metric: str = field(
        default="unexpected_score",
        metadata={
            "choices": ["unexpected_score", "unexpected_and_absq"],
            "help": "How to select the action to reflect on"
        }
    )
    max_reflections_per_task: int = field(
        default=2,
        metadata={"help": "The maximum number of reflections to generate per task."}
    )
    reflection_threshold: float = field(
        default=0.5,
        metadata={"help": "Only reflect on data that exceeds this threshold."}
    )
    min_retrieval_score: float = field(
        default=0.25,
        metadata={"help": "The minimum score to consider a reflection as valid."}
    )
    max_to_retrieve: int = field(
        default=2,
        metadata={"help": "The maximum number of reflections to retrieve from the database."}
    )
    use_gt_success: bool = field(
        default=False,
        metadata={"help": "Whether to use ground truth success of estimated success for EOT reward"}
    )

    def __post_init__(self):
        super().__post_init__()

        _api_base = os.environ.get("REFLECTION_LLM_API_BASE", "")
        assert _api_base != "", "Did you forget to set your REFLECTION_LLM_API_BASE?"
        print(f"Using REFLECTION_LLM_API_BASE: {_api_base}")

        assert os.environ.get("REFLECTION_LLM_API_KEY", "") != "", "Did you forget to set your REFLECTION_LLM_API_KEY?"
        return


class RMCTSRenderHelper(MCTSRenderHelper):
    """helps render the RMCTS search tree for debugging/analysis
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

            ### TODO: format different value function stuff here
            additional_value_info_str = ""

            reflection_str = "Retrieved Reflections=\l"
            retr_reflections = state._additional_info.get('retrieved_policy_reflections', [])
            for r_idx, r in enumerate(retr_reflections):
                r: dict # simplify reflection info
                reflection_str += fr"[IDX={r_idx}]:{r.get('reflection', '')}\l"
            reflection_str = break_long_string(reflection_str).strip().replace('\n', '\l')

            label = fr"{reflection_str}\l======\l" + fr"{value_str}\l" + fr"{additional_value_info_str}\l======\l" + fr"{act_str}\l" + fr"{reason_str}\l"
            graph.edge(f'{seen_nodes[node_hash]}', f'{seen_nodes[child_hash]}', label=label)
        return


class RMCTSAgent(MCTSAgent, RAgentMixin):
    name: str = "rmcts"

    def __init__(
        self,
        args: RMCTSAgentArgs,
        value_function: ValueFunction,
        action_space="computer_13",
        observation_type="screenshot_a11y_tree",
        platform="ubuntu"
    ):
        MCTSAgent.__init__(self, args, value_function, action_space, observation_type, platform)
        RAgentMixin.__init__(self, args.db_path)
        self.args = args

        self._task_record_folder_path = os.path.join(self.db_path, "task_records")
        if not os.path.exists(self._task_record_folder_path):
            os.makedirs(self._task_record_folder_path, exist_ok=True)

        self.policy_prompt = ReinforcedReACTPolicy(
            args=ReinforcedReACTPolicyArgs(
                max_trajectory_length=self.args.max_trajectory_length,
                embedding_model=self.args.embedding_model,
                embedding_api_provider=self.args.embedding_api_provider,
                rlm_model=self.args.rlm_model,
                rlm_api_provider=self.args.rlm_api_provider,
                rlm_temperature=self.args.rlm_temperature,
                rlm_top_p=self.args.rlm_top_p,
                rlm_max_tokens=self.args.rlm_max_tokens,
                rlm_force_context_truncation=self.args.rlm_force_context_truncation,
                rlm_flatten_chat_msg=self.args.rlm_flatten_chat_msg,
                rlm_flatten_engine=self.args.rlm_flatten_engine,
                rlm_max_context_length=self.args.rlm_max_context_length,
                db_path=args.db_path,
                max_reflections_per_task=self.args.max_reflections_per_task,
                reflection_threshold=self.args.reflection_threshold,
                min_retrieval_score=self.args.min_retrieval_score,
                max_to_retrieve=self.args.max_to_retrieve,
                use_gt_success=self.args.use_gt_success,
            ),
            system_message=self.system_message,
            observation_type=self.observation_type,
            action_space=self.action_space,
            policy_lm_config=self.lm_config,
        )
        
        ### used by visualization later
        self._policy_metadata = {}
        self._value_metadata = {}
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

    @time_it
    def on_task_start(self, task_info: dict, **kwargs) -> None:
        prompt_constructor: ReinforcedPolicyMixin = self.policy_prompt
        prompt_constructor.on_task_start(task_info)

        value_function: ReinforcedValueFunctionMixin = self.value_function
        value_function.on_task_start(task_info)
        return
    
    def _expansion(self, state: Node):
        super(RMCTSAgent, self)._expansion(state)
        
        ### save the retrieved reflections into metadata
        prompt_constructor: ReinforcedPolicyMixin = self.policy_prompt
        curr_state_obs = state.observations[-1]
        instruction = state._additional_info['instruction']

        # this should be cached already
        retrieval_key = get_rpolicy_retrieval_key(instruction, curr_state_obs)
        retrieved_reflections = prompt_constructor._retrieval_cache.get(retrieval_key, [])
        simpl_reflections = [r.simplified_info() for r in retrieved_reflections]
        
        state._additional_info['retrieved_policy_reflections'] = simpl_reflections
        return

    def _get_branch_stats(self, root_node: Node, response_seqs: list[str]) -> dict[str, list]:
        found_qs = []
        found_Nsa = []
        found_P = []
        found_V_next = []

        curr_node = root_node
        for resp in response_seqs:
            assert resp in curr_node.children, f"Response {resp} not found in {curr_node.children.keys()}"

            curr_state_str_hash = curr_node._to_string_rep()
            found_qs.append(self.Q[curr_state_str_hash][resp])
            found_Nsa.append(self.Nsa[curr_state_str_hash][resp])
            found_P.append(self.P[curr_state_str_hash][resp])

            next_node = curr_node.children[resp]
            found_V_next.append(next_node.value)

            curr_node = next_node
        return {
            'Q': found_qs,
            'Nsa': found_Nsa,
            'P': found_P,
            'V_next': found_V_next
        }

    def _get_response_based_trajectory(self, full_trajectory: list[dict]):
        ## by default, the outer running script appends per step based on ACTION instead of response
        ## but we can distinguish it using the "step_idx": step_idx
        response_based_trajectory = []
        step_idx_to_actions = {}
        curr_step_idx = 0
        for i, data in enumerate(full_trajectory):
            if 'step_idx' in data:
                step_idx = data['step_idx']
                if step_idx == curr_step_idx:
                    ### first time encouter, add (s, a) pair
                    raw_obs = full_trajectory[i-1]['obs']
                    obs = self.obs_processor(raw_obs)
                    response_based_trajectory.append(obs)
                    response_based_trajectory.append(data)
                    curr_step_idx += 1
                
                action = data['action']
                if step_idx not in step_idx_to_actions:
                    step_idx_to_actions[step_idx] = []
                step_idx_to_actions[step_idx].append(action)
        # add last state
        assert 'obs' in full_trajectory[-1], "last data in trajectory should be a state"
        raw_obs = full_trajectory[-1]['obs']
        obs = self.obs_processor(raw_obs)
        response_based_trajectory.append(obs)

        # add action lists
        for data in response_based_trajectory:
            if 'step_idx' in data:
                step_idx = data['step_idx']
                data['actions'] = step_idx_to_actions[step_idx]
        return response_based_trajectory

    @time_it
    def on_task_end(
        self,
        actual_trajectory: list[dict],
        task_info: dict,
        meta_data: Any,
        **kwargs
    ) -> None:
        prompt_constructor: ReinforcedPolicyMixin = self.policy_prompt
        value_function: ReinforcedValueFunctionMixin = self.value_function

        resp_based_traj = self._get_response_based_trajectory(actual_trajectory)

        all_responses: list[str] = []
        for data in resp_based_traj:
            if 'raw_action' in data:
                all_responses.append(data['raw_action'])
        
        search_tree_stats = self._get_branch_stats(self.root_node, all_responses)
        
        Q = search_tree_stats['Q']
        mean_Q = np.mean([q for q in Q if q is not None])
        new_task_record = TaskRecord(
            trajectory=resp_based_traj,
            task_info=task_info,
            Q=search_tree_stats['Q'],
            Nsa=search_tree_stats['Nsa'],
            P=search_tree_stats['P'],
            V_next=search_tree_stats['V_next'],
            success=meta_data['success'],
            est_success=1.0 if mean_Q > 0.7 else 0.0,
        )
        value_function.on_task_end(resp_based_traj, task_info, meta_data, task_record=new_task_record)
        prompt_constructor.on_task_end(resp_based_traj, task_info, meta_data, task_record=new_task_record)

        ### save the task record
        # assuming that on_task_end fills in new stuff in the task record, if any
        all_task_records: list[TaskRecord] = self._load_lzma_db_files(self._task_record_folder_path)
        logger.info(f"Found {len(all_task_records)} task records from {self._task_record_folder_path}")
        existing_task_hashes = set([hash(r) for r in all_task_records])
        new_task_record_to_write = [r for r in [new_task_record] if hash(r) not in existing_task_hashes]
        logger.info(f"Deduped and writing {len(new_task_record_to_write)} new task records to {self._task_record_folder_path}")
        self._write_lzma_db_files(self._task_record_folder_path, new_task_record_to_write)
        return

    def save_search_tree(self, root_state: Node, traversal_cache: list, result_dir: str):
        # same code but using a different renderer
        #### for later visualization
        logger.info("Saving search tree visualization")
        save_dir = os.path.join(result_dir, "search_viz")
        render_helper = RMCTSRenderHelper(
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
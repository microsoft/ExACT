import os
import time
import pickle
import lzma
import logging
import copy
import math
from typing import Any
from typing import Optional
from PIL import Image
from browser_env import Trajectory, ActionTypes
from browser_env.utils import pil_to_b64
from src.logging import time_it
from src.agentic.value_function import RubricBasedValueFunctionMixin, DebateBasedValueFunctionMixin
from src.agentic.rvalue_function import ReinforcedValueFunctionMixin
from src.agent.agent_args import AgentArguments
from src.agentic.rpolicy import ReinforcedPromptMixin
from src.agentic.rvalue_function import _pil_image_to_str
from src.envs.actions import ActionTypes, Action
from src.agent.mcts_agent import (
    MCTSAgent, MCTSRenderHelper, MCTSwDBTRenderHelper, Node,
    break_long_string
)
from graphviz import Digraph


logger = logging.getLogger('rmcts_agent')


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
            qsa = Q[node_hash].get(a, a.metadata.get('Q', 0.0))

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

            reflection_str = "Retrieved Reflections=\l"
            retr_reflections = a.metadata.get('retrieved_reflections', [])
            for r_idx, r in enumerate(retr_reflections):
                r: dict # simplify reflection info
                reflection_str += fr"[IDX={r_idx}]:{r.get('reflection', '')}\l"
            reflection_str = break_long_string(reflection_str).strip().replace('\n', '\l')

            label = fr"{reflection_str}\l======\l"+ fr"{value_str}\l" + fr"{act_str}\l" + fr"{reason_str}\l"
            graph.edge(f'{seen_nodes[node_hash]}', f'{seen_nodes[child_hash]}', label=label)
        return


class RMCTSAgent(MCTSAgent):
    @time_it
    def on_task_start(self, task_info: dict, **kwargs) -> None:
        prompt_constructor: ReinforcedPromptMixin = self.prompt_constructor
        prompt_constructor.on_task_start(task_info)

        value_function: ReinforcedValueFunctionMixin = self.value_function
        value_function.on_task_start(task_info)
        return
    
    def _gen_next_actions(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
    ) -> list[Action]:
        actions = super(RMCTSAgent, self)._gen_next_actions(trajectory, intent, meta_data, images)
        
        ### save the retrieved reflections into metadata
        prompt_constructor: ReinforcedPromptMixin = self.prompt_constructor
        state_info = trajectory[-1]
        curr_obs= state_info["observation"]
        # this should be cached already
        retrieved_reflections = prompt_constructor._retrieval_cache.get((intent, curr_obs["text"]), [])
        simpl_reflections = [r.simplified_info() for r in retrieved_reflections]
        for a in actions:
            a.metadata['retrieved_reflections'] = simpl_reflections
        return actions

    async def _simulation(self, state: Node) -> float:
        ### return value by (optionally) doing roll out
        logger.info("Simulating state value")
        if state.is_root:
            return 0.0
        
        v_model = self.value_function_model
        intent = state._additional_info['intent']
        images = state._additional_info.get('images', [])  # task intent images
        init_screenshot = Image.fromarray(state.trajectory[0]["observation"]["image"])
        all_screenshots = []
        for data in state.trajectory:
            if isinstance(data, dict):
                all_screenshots.append(Image.fromarray(data["observation"]["image"]))
        all_actions_str = state._additional_info['meta_data']['action_history']
        if all_actions_str[0].lower() == "none":
            all_actions_str = all_actions_str[1:]  # vfunc expects (s,a,s,...). Remove the none padding
        logger.debug(f"all_actions_str has {all_actions_str}")
        try:
            # evaluate current state
            # last_screenshots = screenshots SINCE the root state
            if self.value_function_model in ["gpt4o"]:
                v = self.value_function.evaluate_success(
                    screenshots=all_screenshots,
                    actions=copy.deepcopy(all_actions_str),
                    current_url=state.env.page.url,
                    last_reasoning="",
                    intent=intent,
                    models=["gpt-4o-2024-05-13"],
                    init_screenshot=init_screenshot,
                    intent_images=images if len(images) > 0 else None
                )
            elif self.value_function_model in ["gpt-4o", "gpt-4o-mini", "OpenGVLab/InternVL2-Llama3-76B"]:
                v = self.value_function.evaluate_success(
                    screenshots=all_screenshots,
                    actions=copy.deepcopy(all_actions_str),
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

    @time_it
    def on_task_end(self, trajectory: Trajectory, score: float, task_info: dict, meta_data: Any, **kwargs) -> None:
        prompt_constructor: ReinforcedPromptMixin = self.prompt_constructor
        value_function: ReinforcedValueFunctionMixin = self.value_function

        all_actions: list[Action] = []
        for data in trajectory:
            if isinstance(data, Action):
                if 'Early stop' in data.answer:
                    continue
                all_actions.append(data)
        
        search_tree_stats = {
            'Q': [a.metadata.get('Q', 0.0) for a in all_actions],
            'Nsa': [a.metadata.get('Nsa', 0.0) for a in all_actions],
            'P': [a.metadata.get('P', 0.0) for a in all_actions],
            'V_next': [a.metadata.get('V_next', 0.0) for a in all_actions],
        }

        # value function needs to go first in order to save the rubrics to task records
        value_function.on_task_end(trajectory, score, task_info, meta_data, search_tree_stats=search_tree_stats)
        prompt_constructor.on_task_end(trajectory, score, task_info, meta_data, search_tree_stats=search_tree_stats)
        return

    def _save_tree_from_root(
        self,
        root_state: Node,
        task_id: int,
        result_dir: str
    ):
        removed_node_data = {
            root_state: {
                'env': root_state.env,
                'action_trajectory': root_state.action_trajectory,
                '_additional_info': root_state._additional_info
            }
        }
        root_state.env = None
        root_state.action_trajectory = None
        root_state._additional_info = None
        all_a_s = root_state._get_all_child_actions()
        for action, next_state in all_a_s:
            # remove data that cannot be pickled
            # clean the trajectory as theres a lot of duplicates
            removed_node_data[next_state] = {
                'env': next_state.env,
                'trajectory': next_state.trajectory,
                'action_trajectory': next_state.action_trajectory,
                '_additional_info': next_state._additional_info
            }
            if next_state._lazy_initialized or action["action_type"] in [ActionTypes.STOP, ActionTypes.NONE]:
                new_tracj = []
            else:
                # only save current observation to save space
                new_tracj = [next_state.trajectory[-1]]

            next_state.env = None
            next_state.trajectory = new_tracj
            next_state.action_trajectory = None
            next_state._additional_info = None
        # save
        curr_time = time.strftime("%Y%m%d-%H%M%S")
        save_file = os.path.join(
            result_dir,
            "search_trees",
            f"task_{task_id}", f"tree_{curr_time}.pkl.xz"
        )
        with lzma.open(save_file, "wb") as fwrite:
            pickle.dump(root_state, fwrite)
        # restore
        root_state.env = removed_node_data[root_state]['env']
        root_state.action_trajectory = removed_node_data[root_state]['action_trajectory']
        root_state._additional_info = removed_node_data[root_state]['_additional_info']
        for action, next_state in all_a_s:
            next_state.env = removed_node_data[next_state]['env']
            next_state.trajectory = removed_node_data[next_state]['trajectory']
            next_state.action_trajectory = removed_node_data[next_state]['action_trajectory']
            next_state._additional_info = removed_node_data[next_state]['_additional_info']
        return

    def save_search_tree(self, root_state: Node, task_info: dict, result_dir: str):
        # same code but using a different renderer
        #### for later visualization
        logger.debug("Saving search tree visualization")
        task_id = task_info["task_id"]
        save_dir = os.path.join(result_dir, "search_trees", f"task_{task_id}")
        render_helper = RMCTSRenderHelper(
            save_dir,
            os.path.join(result_dir, "tmp_images")
        )
        render_helper.render(root_state, self.Q)

        #### save rubrics used
        v_func: ReinforcedValueFunctionMixin | RubricBasedValueFunctionMixin = self.value_function
        intent = task_info['intent']
        images = task_info.get('images', [])
        init_screenshot_array = root_state.trajectory[0]["observation"]["image"]
        init_screenshot = Image.fromarray(init_screenshot_array)
        encoded_image_str = _pil_image_to_str(images + [init_screenshot])

        rubrics_used = v_func._rubrics_cache.get((intent, encoded_image_str), "")
        rubric_reflections_used = v_func._retrieval_cache.get((intent, encoded_image_str), [])
        rubrics_refl_simplified = []
        for rub_r in rubric_reflections_used:
            rubrics_refl_simplified.append(rub_r.reflection)
        rubrics_refl_simplified_str = "\n".join(rubrics_refl_simplified)
        save_file = os.path.join(result_dir, "search_trees", f"task_{task_id}", f"rubrics.txt")
        with open(save_file, "w") as fwrite:
            fwrite.write(f"Retrieved Reflections:\n{rubrics_refl_simplified_str}\n\nRubrics:\n{rubrics_used}")

        ### maybe for later training
        logger.debug("Saving raw search tree data")
        ### maybe for later training
        self._save_tree_from_root(root_state, task_id, result_dir)
        return


class RMCTSwDBTRenderHelper(MCTSwDBTRenderHelper):
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
            qsa = Q[node_hash].get(a, a.metadata.get('Q', 0.0))

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

            reflection_str = "Retrieved Reflections=\l"
            retr_reflections = a.metadata.get('retrieved_reflections', [])
            for r_idx, r in enumerate(retr_reflections):
                r: dict # simplify reflection info
                reflection_str += fr"[IDX={r_idx}]:{r.get('reflection', '')}\l"
            reflection_str = break_long_string(reflection_str).strip().replace('\n', '\l')

            label = fr"{reflection_str}\l======\l" + fr"{value_str}\l" + fr"{debate_str}\l======\l" + fr"{act_str}\l" + fr"{reason_str}\l"
            graph.edge(f'{seen_nodes[node_hash]}', f'{seen_nodes[child_hash]}', label=label)
        return


class RMCTSwDBTAgent(RMCTSAgent):
    async def _simulation(self, state: Node) -> float:
        ### return value by (optionally) doing roll out
        logger.info("Simulating state value")
        if state.is_root:
            return 0.0
        
        v_model = self.value_function_model
        intent = state._additional_info['intent']
        images = state._additional_info.get('images', [])  # task intent images
        init_screenshot = Image.fromarray(state.trajectory[0]["observation"]["image"])
        all_screenshots = []
        all_screenshots_text = []
        for data in state.trajectory:
            if isinstance(data, dict):
                all_screenshots.append(Image.fromarray(data["observation"]["image"]))
                all_screenshots_text.append(data["observation"]["text"])
        all_actions_str = state._additional_info['meta_data']['action_history']
        if all_actions_str[0].lower() == "none":
            all_actions_str = all_actions_str[1:]  # vfunc expects (s,a,s,...). Remove the none padding
        logger.debug(f"all_actions_str has {all_actions_str}")
        try:
            # evaluate current state
            # last_screenshots = screenshots SINCE the root state
            if self.value_function_model in ["gpt4o"]:
                v = self.value_function.evaluate_success(
                    screenshots=all_screenshots,
                    screenshots_text=all_screenshots_text,  # newly added for reinforced debate value function
                    actions=copy.deepcopy(all_actions_str),
                    current_url=state.env.page.url,
                    last_reasoning="",
                    intent=intent,
                    models=["gpt-4o-2024-05-13"],
                    init_screenshot=init_screenshot,
                    intent_images=images if len(images) > 0 else None
                )
            elif self.value_function_model in ["gpt-4o", "gpt-4o-mini", "OpenGVLab/InternVL2-Llama3-76B"]:
                v = self.value_function.evaluate_success(
                    screenshots=all_screenshots,
                    screenshots_text=all_screenshots_text,  # newly added for reinforced debate value function
                    actions=copy.deepcopy(all_actions_str),
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

        # also save the supporting and opposing arguments
        v_func: DebateBasedValueFunctionMixin = self.value_function
        v_func_data_key = v_func._encode_eval_success_input(
            all_screenshots,
            copy.deepcopy(all_actions_str),
            intent,
            images if len(images) > 0 else []
        )
        v_func_data = v_func._debate_cache.get(v_func_data_key, {})
        state._additional_info['debate_data'] = v_func_data  # may be used later for rendering
        logger.debug(f"Loaded debate data={v_func_data}")
        return v

    async def search(self, state: Node):
        """perform one iteration of MCTS: selection, expansion, simulation, backpropagation
        # overriding the base MCTS class because we need to store the debate data into action
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
        best_action.metadata["next_V_debate_data"] = next_state._additional_info.get('debate_data', {})

        if v == 1.0:
            best_action_strs = [a.to_short_str() for a in self.best_action_cache]
            best_action_str = "\n--->\n".join(best_action_strs)
            logger.info(f"found task finishing trajectory: {best_action_str}")
            self.found_success_trajectory = True
        return v
    
    async def anext_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        additional_inputs: dict[str, Any],
    ) -> Action:
        # add puct to be an argument
        cmd_args: AgentArguments = additional_inputs["cmd_args"]
        self.puct = cmd_args.puct
        logger.debug(f"PUCT value set to {self.puct}")

        final_action = await super().anext_action(trajectory, intent, meta_data, additional_inputs)
        return final_action

    @time_it
    def on_task_end(self, trajectory: Trajectory, score: float, task_info: dict, meta_data: Any, **kwargs) -> None:
        prompt_constructor: ReinforcedPromptMixin = self.prompt_constructor
        value_function: ReinforcedValueFunctionMixin = self.value_function

        all_actions: list[Action] = []
        all_debate_data = []
        for data in trajectory:
            if isinstance(data, Action):
                if 'Early stop' in data.answer:
                    continue
                all_actions.append(data)
                all_debate_data.append(data.metadata.get('next_V_debate_data', {}))
        
        search_tree_stats = {
            'Q': [a.metadata.get('Q', 0.0) for a in all_actions],
            'Nsa': [a.metadata.get('Nsa', 0.0) for a in all_actions],
            'P': [a.metadata.get('P', 0.0) for a in all_actions],
            'V_next': [a.metadata.get('V_next', 0.0) for a in all_actions],
        }

        # value function needs to go first in order to save the rubrics to task records
        value_function.on_task_end(trajectory, score, task_info, meta_data, search_tree_stats=search_tree_stats, debate_data=all_debate_data)
        prompt_constructor.on_task_end(trajectory, score, task_info, meta_data, search_tree_stats=search_tree_stats)
        return
    
    def save_search_tree(self, root_state: Node, task_info: dict, result_dir: str):
        # same code but using a different renderer
        #### for later visualization
        logger.debug("Saving search tree visualization")
        task_id = task_info["task_id"]
        save_dir = os.path.join(result_dir, "search_trees", f"task_{task_id}")
        render_helper = RMCTSwDBTRenderHelper(
            save_dir,
            os.path.join(result_dir, "tmp_images")
        )
        render_helper.render(root_state, self.Q)

        ### maybe for later training
        logger.debug("Saving raw search tree data")
        ### maybe for later training
        self._save_tree_from_root(root_state, task_id, result_dir)
        return
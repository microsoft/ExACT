from typing import Any, Optional
from exact.env.desktop_env_dev import DynamicPooledDesktopEnv
from exact.logging import time_it
import logging
import copy


logger = logging.getLogger("src.env.desktop_env_resettable")


class ResettableDesktopEnv(DynamicPooledDesktopEnv):
    """Environment where step() supports UNDO action. This is useful for ExACT type agents.
    Note: due to how its implemented, you should NOT call the following function externally:
    - simu_step
    - simu_evaluate
    - simu_reset
    
    because these are used internally to implement the UNDO action.
    """
    def __init__(
        self,
        provider_name: str = "vmware",
        region: str = None,
        path_to_vm: str = None,
        snapshot_name: str = "init_state",
        action_space: str = "computer_13",
        cache_dir: str = "cache",
        screen_size: tuple[int] = (1920, 1080),
        headless: bool = False,
        require_a11y_tree: bool = True,
        require_terminal: bool = False,
        n_instances_per_task: int = 1,
        os_type: str = "Ubuntu",
    ):
        super().__init__(
            provider_name=provider_name,
            region=region,
            path_to_vm=path_to_vm,
            snapshot_name=snapshot_name,
            action_space=action_space,
            cache_dir=cache_dir,
            screen_size=screen_size,
            headless=headless,
            require_a11y_tree=require_a11y_tree,
            require_terminal=require_terminal,
            n_instances_per_task=n_instances_per_task,
            os_type=os_type
        )
        
        self.task_config = None
        self.response_trajectory = [] # list of str
        self.action_trajectory = []  # list of list
        self._curr_env_idx = -100  # reserved env
        self._curr_step_idx = -1
        self._resp_hist_to_env_idx = {}  # always full actions given a response
        self._env_idx_to_obs = {}  # always full actions given a response
        self.__obs_cache = {}  # obs corresponding to executing self.response_trajectory
        
        ### debugging
        self.__full_response_trajectory = []
        return
    
    @time_it
    def reset(self, task_config: Optional[dict[str, Any]] = None, seed=None, options=None) -> dict[str, Any]:
        self.task_config = task_config
        
        self.response_trajectory = []
        self.action_trajectory = []
        self._curr_env_idx = -100
        self._curr_step_idx = -1
        self._resp_hist_to_env_idx = {}
        self._env_idx_to_obs = {}
        self.__obs_cache = {}
        self.__full_response_trajectory = []
        
        observation = super().reset(task_config=task_config, seed=seed, options=options)
        self.__obs_cache = {
            'obs': observation,
            'reward': 0.0,
            'done': False,
            'info': None
        }
        return observation
    
    def _update_resp_hist_to_env_idx(self, response_trajectory, env_idx):
        env_key = tuple(response_trajectory)
        _resp_hist_to_env_idx_copy = copy.deepcopy(self._resp_hist_to_env_idx)
        for k, v in _resp_hist_to_env_idx_copy.items():
            if v == env_idx:
                self._resp_hist_to_env_idx.pop(k)
        
        self._resp_hist_to_env_idx[env_key] = env_idx
        return
    
    def get_curr_response_trajectory(self):
        # used for debugging purposes, or for wrapper classes
        return self.response_trajectory
    
    def get_full_response_trajectory(self):
        # used for debugging purposes, or for wrapper classes
        return self.__full_response_trajectory
    
    @time_it
    def step(self, action, resp_str: str, step_idx: int, pause=0.5):
        """step that supports UNDO action. Note that this could take a LONG time to run, as it may need to replay many actions.
        """
        logger.debug(f"calling step with {step_idx=}, {action=}. Currently {self._curr_step_idx=}, {self.response_trajectory=}, {self.action_trajectory=}")
        
        # new step, store outputs related to self.response_trajectory so far
        if step_idx != self._curr_step_idx:
            self._update_resp_hist_to_env_idx(self.response_trajectory, self._curr_env_idx)
            self._env_idx_to_obs[self._curr_env_idx] = {
                'obs': self.__obs_cache['obs'],
                'reward': self.__obs_cache['reward'],
                'done': self.__obs_cache['done'],
                'info': self.__obs_cache['info']
            }
            
        if action == "UNDO":
            # need to change env
            if len(self.response_trajectory) == 0:
                logger.warning("Received UNDO but no actions to undo. Doing no-op here.")
            else:
                self.response_trajectory.pop()
                self.action_trajectory.pop()
            
            self.__full_response_trajectory.append(resp_str)
            
            env_idx = self._get_env_idx_for_resp_traj(
                response_trajectory=self.response_trajectory,
                actions_trajectory=self.action_trajectory,
                sleep_after_execution=pause
            )
            self._curr_env_idx = env_idx
            
            logger.info(f"UNDO to state {self.response_trajectory=}, {self.action_trajectory=}, {env_idx=}")
            state_data = self._env_idx_to_obs[env_idx]
            obs = state_data['obs']
            reward = state_data['reward']
            done = state_data['done']
            info = state_data['info']
        else:
            # continue from current env
            # new step
            if step_idx != self._curr_step_idx:
                self.response_trajectory.append(resp_str)
                self.action_trajectory.append([action])
                
                self.__full_response_trajectory.append(resp_str)
            # old step
            else:
                self.action_trajectory[-1].append(action)
            
            if self._curr_env_idx == -100:
                obs, reward, done, info = self.reserved_env.step(action, pause)
            else:
                obs, reward, done, info = self.simu_step(action, env_idx=self._curr_env_idx, pause=pause)
        
        self._curr_step_idx = step_idx
        self.__obs_cache = {
            'obs': obs,
            'reward': reward,
            'done': done,
            'info': info
        }
        logger.debug(f"step {self._curr_step_idx=}, {self.__full_response_trajectory=}, {self.response_trajectory=}")
        return obs, reward, done, info
    
    def _get_env_idx_for_resp_traj(
        self,
        response_trajectory: list[str],
        actions_trajectory: list[list[str]],
        sleep_after_execution: float
    ) -> int:
        # find state that corresponds to executing all actions in response_trajectory
        # note that self._env_idx_to_obs should always assign to full response_trajectory, not by actions
        task_config = self.task_config
        
        if len(response_trajectory) == 0:
            env_idx = self._get_unused_env_ids(task_config, create_new_if_empty=True)[0]
            obs, reward, done, info = self.simu_step(
                'WAIT',
                env_idx=env_idx,
                pause=sleep_after_execution
            )
            self._env_idx_to_obs[env_idx] = {
                'obs': obs,
                'reward': reward,
                'done': done,
                'info': info
            }
        else:
            logger.debug(f"Finding env corresponding to {response_trajectory}")
            curr_responses = response_trajectory
            curr_actions = actions_trajectory
            env_key = tuple(response_trajectory)

            logger.debug(f"Current cache: {self._resp_hist_to_env_idx}")
            _responses_to_replay = []   # for debugging
            _nested_actions_to_replay = []  # NOT nested list
            
            while True:
                if len(curr_responses) == 0:
                    env_idx = self._get_unused_env_ids(task_config, create_new_if_empty=True)[0]
                    logger.debug(f"Go back to empty parent. Starting anew with {env_idx=}")
                    break
                if env_key in self._resp_hist_to_env_idx:
                    env_idx = self._resp_hist_to_env_idx[env_key]
                    logger.debug(f"Trajectory {curr_responses} found in cache. Returning with {env_idx=}")
                    assert env_idx in self._env_idx_to_obs, f"{env_idx=} not in {self._env_idx_to_obs=}"
                    return env_idx
                
                logger.debug(f"Trajectory {curr_responses} not found in cache. Backing off one response.")
                ## replay this response
                _r_to_replay = curr_responses[-1]
                _responses_to_replay.append(_r_to_replay)
                a_to_replay = curr_actions[-1]
                _nested_actions_to_replay.append(a_to_replay)

                ## and go back
                curr_responses = curr_responses[:-1]
                curr_actions = curr_actions[:-1]
                env_key = tuple(curr_responses)
            
            _responses_to_replay = _responses_to_replay[::-1]  # real sequence
            _nested_actions_to_replay = _nested_actions_to_replay[::-1]
            logger.debug(f"Replaying responses: {_responses_to_replay}")
            logger.debug(f"Replaying actions: {_nested_actions_to_replay}")

            for actions in _nested_actions_to_replay:
                for a in actions:
                    obs, reward, done, info = self.simu_step(
                        a,
                        env_idx=env_idx,
                        pause=sleep_after_execution
                    )
                    logger.debug(f'received {info=}, {reward=}, {done=}')
                    if done:
                        logger.info(f"The episode is done after replaying actions: {actions}")
                        break
            self._env_idx_to_obs[env_idx] = {
                'obs': obs,
                'reward': reward,
                'done': done,
                'info': info
            }
        return env_idx
    
    def evaluate(self):
        logger.debug(f"evaluate {self._curr_step_idx=}, {self._curr_env_idx=}, {self.__full_response_trajectory=}, {self.response_trajectory=}")
        if self._curr_env_idx == -100:
            return self.reserved_env.evaluate()
        else:
            return self.simu_evaluate(env_idx=self._curr_env_idx)
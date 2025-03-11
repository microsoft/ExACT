from __future__ import annotations

import logging
import os
import time
import shutil
from typing import Callable, Any, Optional, Tuple
from typing import List, Dict, Union

import gymnasium as gym

from desktop_env.controllers.python import PythonController
from desktop_env.controllers.setup import SetupController
from desktop_env.evaluators import metrics, getters
from desktop_env.providers import create_vm_manager_and_provider
from exact.logging import time_it
from exact.env.utils import timeout, retry_timeout


logger = logging.getLogger("desktopenv.env")

Metric = Callable[[Any, Any], float]
Getter = Callable[[gym.Env, Dict[str, Any]], Any]


class DesktopEnv(gym.Env):
    """
    DesktopEnv with OpenAI Gym interface. It provides a desktop environment for setting and evaluating desktop automation tasks.
    """
    @time_it
    def __init__(
            self,
            provider_name: str = "vmware",
            region: str = None,
            path_to_vm: str = None,
            snapshot_name: str = "init_state",
            action_space: str = "computer_13",
            cache_dir: str = "cache",
            screen_size: Tuple[int] = (1920, 1080),
            headless: bool = False,
            require_a11y_tree: bool = True,
            require_terminal: bool = False,
            os_type: str = "Ubuntu",
    ):
        """
        Args:
            provider_name (str): virtualization provider name, default to "vmware"
            region (str): the region for allocate machines, work for cloud services, default to  "us-east-1"
            path_to_vm (str): path to .vmx file
            snapshot_name (str): snapshot name to revert to, default to "init_state"
            action_space (str): "computer_13" | "pyautogui"
            cache_dir (str): cache directory to cache task-related stuffs like
              reference file for evaluation
            screen_size (Tuple[int]): screen size of the VM
            headless (bool): whether to run the VM in headless mode
            require_a11y_tree (bool): whether to require accessibility tree
            require_terminal (bool): whether to require terminal output
        """
        # Initialize VM manager and vitualization provider
        self.region = region

        # Default
        self.server_port = 5000
        self.chromium_port = 9222
        self.vnc_port = 8006
        self.vlc_port = 8080
        self.manager, self.provider = create_vm_manager_and_provider(provider_name, region)

        self.os_type = os_type

        # Initialize environment variables
        if path_to_vm:
            self.path_to_vm = os.path.abspath(os.path.expandvars(os.path.expanduser(path_to_vm))) \
                if provider_name in {"vmware", "virtualbox"} else path_to_vm
        else:
            self.path_to_vm = self.manager.get_vm_path(self.os_type, region)

        self.snapshot_name = snapshot_name
        self.cache_dir_base: str = cache_dir
        # todo: add the logic to get the screen size from the VM
        self.headless = headless
        self.require_a11y_tree = require_a11y_tree
        self.require_terminal = require_terminal

        # Initialize emulator and controller
        if provider_name != "docker": # Check if this is applicable to other VM providers
            logger.info("Initializing...")
            self._start_emulator()

        # mode: human or machine
        self.instruction = None
        assert action_space in ["computer_13", "pyautogui"]
        self.action_space = action_space  # todo: refactor it to the ActType

        # episodic stuffs, like counters, will be updated or reset
        # when calling self.reset()
        self._traj_no: int = -1
        self._step_no: int = 0
        self.action_history: List[Dict[str, any]] = []
        self._is_docker_running = False
        return

    @time_it
    def _pause_emulator(self):
        logger.info("Pausing emulator...")
        self.provider.pause_emulator(self.path_to_vm)
        return

    @time_it
    def _unpause_emulator(self):
        logger.info("Unpausing emulator...")
        self.provider.unpause_emulator(self.path_to_vm)
        return

    @time_it
    def _start_emulator(self):
        # Power on the virtual machine
        # assumes DockerProvider (which is modified to take in these ports for parallelization)
        self.provider.start_emulator(
            self.path_to_vm, self.headless, self.os_type,
            vnc_port=self.vnc_port,
            server_port=self.server_port,
            chromium_port=self.chromium_port,
            vlc_port=self.vlc_port
        )

        # Get the ip from the virtual machine, and setup the controller
        vm_ip_ports = self.provider.get_ip_address(self.path_to_vm).split(':')
        self.vm_ip = vm_ip_ports[0]
        if len(vm_ip_ports) > 1:
            self.server_port = int(vm_ip_ports[1])
            self.chromium_port = int(vm_ip_ports[2])
            self.vnc_port = int(vm_ip_ports[3])
            self.vlc_port = int(vm_ip_ports[4])
        self.controller = PythonController(vm_ip=self.vm_ip, server_port=self.server_port)
        self.setup_controller = SetupController(
            vm_ip=self.vm_ip,
            server_port=self.server_port,
            chromium_port=self.chromium_port,
            vlc_port=self.vlc_port,
            cache_dir=self.cache_dir_base
        )
        self._is_docker_running = True
        return

    @time_it
    def _revert_to_snapshot(self):
        # Revert to certain snapshot of the virtual machine, and refresh the path to vm and ip of vm
        # due to the fact it could be changed when implemented by cloud services
        path_to_vm = self.provider.revert_to_snapshot(self.path_to_vm, self.snapshot_name)
        if path_to_vm and not path_to_vm == self.path_to_vm:
            # path_to_vm has to be a new path
            self.manager.delete_vm(self.path_to_vm, self.region)
            self.manager.add_vm(path_to_vm, self.region)
            self.manager.occupy_vm(path_to_vm, os.getpid(), self.region)
            self.path_to_vm = path_to_vm
        return

    @time_it
    def _save_state(self, snapshot_name=None):
        # Save the current virtual machine state to a certain snapshot name
        self.provider.save_state(self.path_to_vm, snapshot_name)
        return

    @time_it
    def close(self):
        # Close (release) the virtual machine
        self.provider.stop_emulator(self.path_to_vm)
        self._is_docker_running = False

        if os.path.exists(self.cache_dir_base):
            shutil.rmtree(self.cache_dir_base)
        return

    @time_it
    def reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        # Reset to certain task in OSWorld
        logger.info("Resetting environment...")
        logger.info("Switching task...")
        logger.info("Setting counters...")
        self._traj_no += 1
        self._step_no = 0
        self.action_history.clear()

        logger.info("Reverting to snapshot to {}...".format(self.snapshot_name))
        self._revert_to_snapshot()
        logger.info(f"Starting emulator... {self._is_docker_running=}")
        if not self._is_docker_running:
            self._start_emulator()
        else:
            self.close()
            self._start_emulator()
        logger.info("Emulator started.")
        self._is_docker_running = True

        if task_config is not None:
            self._set_task_info(task_config)
            self.setup_controller.reset_cache_dir(self.cache_dir)
            logger.info("Setting up environment...")
            self.setup_controller.setup(self.config)
            logger.info("Environment setup complete.")

        time.sleep(5)  # wait for applications to open
        observation = self._get_obs()

        self._pause_emulator()
        return observation

    @time_it
    def thread_safe_reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        # Reset to certain task in OSWorld
        logger.info("Resetting environment...")
        logger.info("Switching task...")
        logger.info("Setting counters...")
        self._traj_no += 1
        self._step_no = 0
        self.action_history.clear()

        logger.info("Reverting to snapshot to {}...".format(self.snapshot_name))
        self._revert_to_snapshot()
        logger.info(f"Starting emulator... {self._is_docker_running=}")
        if not self._is_docker_running:
            self._start_emulator()
        else:
            self.close()
            self._start_emulator()
        logger.info("Emulator started.")
        self._is_docker_running = True

        if task_config is not None:
            self._set_task_info(task_config)
            self.setup_controller.reset_cache_dir(self.cache_dir)
            logger.info("Setting up environment...")
            self.setup_controller.setup(self.config)
            logger.info("Environment setup complete.")

        time.sleep(5)  # wait for applications to open
        observation = self._thread_safe_get_obs()

        self._pause_emulator()
        return observation

    @retry_timeout(3)
    @timeout(10)
    def _get_obs_screenshot(self):
        return self.controller.get_screenshot()

    def _thread_safe_get_obs_screenshot(self):
        return self.controller.get_screenshot()

    @retry_timeout(3)
    @timeout(10)
    def _get_obs_accessibility_tree(self):
        if self.require_a11y_tree:
            return self.controller.get_accessibility_tree()
        else:
            return None

    def _thread_safe_get_obs_accessibility_tree(self):
        if self.require_a11y_tree:
            return self.controller.get_accessibility_tree()
        else:
            return None

    @retry_timeout(3)
    @timeout(10)
    def _get_obs_terminal(self):
        if self.require_terminal:
            return self.controller.get_terminal_output()
        else:
            return None

    def _thread_safe_get_obs_terminal(self):
        if self.require_terminal:
            return self.controller.get_terminal_output()
        else:
            return None
    
    @time_it
    def _get_obs(self):
        # We provide screenshot, accessibility_tree (optional), terminal (optional), and instruction.
        # can be customized and scaled
        return {
            "screenshot": self._get_obs_screenshot(),
            "accessibility_tree": self._get_obs_accessibility_tree(),
            "terminal": self._get_obs_terminal(),
            "instruction": self.instruction
        }

    @time_it
    def _thread_safe_get_obs(self):
        # We provide screenshot, accessibility_tree (optional), terminal (optional), and instruction.
        # can be customized and scaled
        return {
            "screenshot": self._thread_safe_get_obs_screenshot(),
            "accessibility_tree": self._thread_safe_get_obs_accessibility_tree(),
            "terminal": self._thread_safe_get_obs_terminal(),
            "instruction": self.instruction
        }

    @property
    def vm_platform(self):
        return self.controller.get_vm_platform()

    @property
    def vm_screen_size(self):
        return self.controller.get_vm_screen_size()

    @time_it
    def _set_task_info(self, task_config: Dict[str, Any]):
        self.task_id: str = task_config["id"]
        self.cache_dir: str = os.path.join(self.cache_dir_base, self.task_id)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.instruction = task_config["instruction"]
        self.config = task_config["config"] if "config" in task_config else []

        # evaluator dict
        # func -> metric function string, or list of metric function strings
        # conj -> conjunction of multiple metrics if func is a list with length > 1, "and"/"or"
        # result -> result getter config, or list of result getter configs
        # expected (optional) -> expected getter config, or list of expected getter configs
        # options (optional) -> metric options, or list of metric options
        # if func is a str list, then result, expected (if exists), options (if exists) should also be lists of the same length
        # even if one of the metrics does not need expected or options field, it should be included in the list with None
        self.evaluator = task_config["evaluator"]
        self.metric: Metric = [getattr(metrics, func) for func in self.evaluator["func"]] \
            if isinstance(self.evaluator["func"], list) \
            else getattr(metrics, self.evaluator["func"])
        self.metric_conj: str = self.evaluator.get("conj", "and")  # take conjunction of multiple metrics
        if "result" in self.evaluator and len(self.evaluator["result"]) > 0:
            self.result_getter: Getter = [getattr(getters, "get_{:}".format(res["type"])) for res in
                                          self.evaluator["result"]] \
                if isinstance(self.evaluator["result"], list) \
                else getattr(getters, "get_{:}".format(self.evaluator["result"]["type"]))
        else:
            self.result_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None

        if "expected" in self.evaluator and len(self.evaluator["expected"]) > 0:
            self.expected_getter: Getter = [getattr(getters, "get_{:}".format(exp["type"])) if exp else None for exp in
                                            self.evaluator["expected"]] \
                if isinstance(self.evaluator["expected"], list) \
                else getattr(getters, "get_{:}".format(self.evaluator["expected"]["type"]))
        else:
            self.expected_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None
        self.metric_options: Union[List[Dict[str, Any]], Dict[str, Any]] = [opt if opt else {} for opt in
                                                                            self.evaluator["options"]] \
            if isinstance(self.evaluator.get("options", {}), list) \
            else self.evaluator["options"] \
            if "options" in self.evaluator \
            else [{}] * len(self.metric) \
            if isinstance(self.metric, list) \
            else {}

        assert (not isinstance(self.evaluator["func"], list)
                or (len(self.metric) == len(self.result_getter) == len(self.expected_getter) == len(
                    self.metric_options)))
        return

    @time_it
    def step(self, action, pause=0.5):
        self._unpause_emulator()

        self._step_no += 1
        self.action_history.append(action)

        reward = 0  # todo: Define reward calculation for each example
        done = False  # todo: Define episode termination condition for each example
        info = {}

        # handle the special actions
        if action in ['WAIT', 'FAIL', 'DONE'] or (type(action) == dict and action['action_type'] in ['WAIT', 'FAIL', 'DONE']):
            if action == 'WAIT':
                time.sleep(pause)
            elif action == 'FAIL':
                done = True
                info = {"fail": True}
            elif action == 'DONE':
                done = True
                info = {"done": True}

        if self.action_space == "computer_13":
            # the set of all possible actions defined in the action representation
            self.controller.execute_action(action)
        elif self.action_space == "pyautogui":
            if action in ['WAIT', 'FAIL', 'DONE']:
                self.controller.execute_action(action)
            else:
                # the set of all possible python commands insides `pyautogui`
                self.controller.execute_python_command(action)

        time.sleep(pause)  # if no pause, sometimes the screenshot is out of sync v.s. the accessibility tree
        observation = self._get_obs()

        self._pause_emulator()
        return observation, reward, done, info

    @time_it
    def evaluate(self):
        """
        Evaluate whether the task is successfully completed.
        """
        self._unpause_emulator()
        time.sleep(3)

        self.setup_controller.setup(self.evaluator.get("postconfig", []))

        if self.evaluator['func'] == "infeasible":
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 1
            else:
                return 0
        else:
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 0

        if type(self.metric) == list:
            results = []
            for idx, metric in enumerate(self.metric):
                try:
                    config = self.evaluator["result"][idx]
                    result_state = self.result_getter[idx](self, config)
                except FileNotFoundError:
                    logger.error("File not found!")
                    if self.metric_conj == 'and':
                        return 0

                expected = self.evaluator["expected"][idx]
                # expected_state = self.expected_getter[idx](self, expected) if expected else None

                # metric: int = metric(result_state, expected_state,
                #                      **self.metric_options[idx]) if expected_state is not None \
                #     else metric(result_state, **self.metric_options[idx])
                if expected:
                    expected_state = self.expected_getter[idx](self, expected)
                    metric: float = metric(result_state, expected_state, **self.metric_options[idx])
                else:
                    metric: float = metric(result_state, **self.metric_options[idx])

                if self.metric_conj == 'and' and float(metric) == 0.0:
                    return 0
                elif self.metric_conj == 'or' and float(metric) == 1.0:
                    return 1
                else:
                    results.append(metric)
            return sum(results) / len(results) if self.metric_conj == 'and' else max(results)
        else:
            try:
                result_state = self.result_getter(self, self.evaluator["result"])
            except FileNotFoundError:
                logger.error("File not found!")
                return 0

            #### problem: expected_state may still be None even after the getter is called
            # expected_state = self.expected_getter(self, self.evaluator["expected"]) if "expected" in self.evaluator \
            #     else None

            # metric: float = self.metric(result_state, expected_state,
            #                             **self.metric_options) if expected_state is not None \
            #     else self.metric(result_state, **self.metric_options)
            if "expected" in self.evaluator:
                expected_state = self.expected_getter(self, self.evaluator["expected"])
                metric: float = self.metric(result_state, expected_state, **self.metric_options)
            else:
                metric: float = self.metric(result_state, **self.metric_options)

        self._pause_emulator()
        return metric

    @time_it
    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            self._unpause_emulator()
            screenshot = self.controller.get_screenshot()
            self._pause_emulator()
            return screenshot
        else:
            raise ValueError('Unsupported render mode: {}'.format(mode))

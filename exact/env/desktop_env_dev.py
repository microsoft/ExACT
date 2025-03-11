from __future__ import annotations

import logging
import shutil
import os
import time
import requests
import json
import concurrent.futures
import psutil
import math
from typing import Callable, Any, Optional, Tuple
from typing import List, Dict, Union

import gymnasium as gym

from exact.env.desktop_env import DesktopEnv
from exact.logging import time_it


logger = logging.getLogger("desktopenv.env")

Metric = Callable[[Any, Any], float]
Getter = Callable[[gym.Env, Dict[str, Any]], Any]


PORT_LOCK_DIR = ".lock"


def _get_available_port(port: int):
    lock_dir = PORT_LOCK_DIR
    os.makedirs(lock_dir, exist_ok=True)  # this is needed if we are running multiple shells

    while port < 65354:
        time.sleep(0.1)
        lock_fname = os.path.join(lock_dir, f"{port}.lock")
        if port in [conn.laddr.port for conn in psutil.net_connections()]:
            port += 2
            continue
        elif os.path.exists(lock_fname):
            ### can return Unless... the lock is NOT valid
            with open(lock_fname, "r") as fread:
                pid = int(fread.read())
            
            if not psutil.pid_exists(pid):
                # NOT VALID!
                os.remove(lock_fname)
            else:
                port += 2
                continue
        
        with open(lock_fname, "w") as fwrite:
            fwrite.write(str(os.getpid()))
        return port


def _release_port(port: int):
    lock_dir = PORT_LOCK_DIR
    lock_fname = os.path.join(lock_dir, f"{port}.lock")
    if os.path.exists(lock_fname):
        try:
            with open(lock_fname, "r") as fread:
                pid = int(fread.read())
            if pid == os.getpid():
                os.remove(lock_fname)
                logger.info(f'Released port {port}')
        except Exception as e:
            logger.error(f'_release_port got {e}')
            pass
    return


def _get_n_available_ports(n: int, start_port: int):
    ports = []
    for _ in range(n):
        port = _get_available_port(start_port)
        ports.append(port)
        start_port = port + 2
    return ports


class PooledDesktopEnv(gym.Env):
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
        n_instances_per_task: int = 1,
        os_type: str = "Ubuntu",
    ):
        """
        same argument as DesktopEnv, except for n_instances_per_task
            n_instances_per_task (int): how many instances to keep in the pool per task

        On a high level, this is what happens:
        - reset: restart ALL instances in the pool
        - step: only use the reserved instance
        - evaluate: only use the reserved instance
        - close: close ALL instances in the pool
        """
        # Initialize VM manager and vitualization provider
        pool = []
        # used for simulation
        _cache_dirs = []

        ### prepare ports as multithreading will be used later
        self._vnc_ports = _get_n_available_ports(n_instances_per_task+1, 8006)
        self._server_ports = _get_n_available_ports(n_instances_per_task+1, 5000)
        self._chromium_ports = _get_n_available_ports(n_instances_per_task+1, 9222)
        self._vlc_ports = _get_n_available_ports(n_instances_per_task+1, 8080)
        for idx in range(n_instances_per_task+1):
            _cache_dir = os.path.join(cache_dir, f"env_{idx}")
            inner_env = DesktopEnv(
                provider_name=provider_name,
                region=region,
                path_to_vm=path_to_vm,
                snapshot_name=snapshot_name,
                action_space=action_space,
                cache_dir=_cache_dir,
                screen_size=screen_size,
                headless=headless,
                require_a11y_tree=require_a11y_tree,
                require_terminal=require_terminal,
                os_type=os_type
            )
            inner_env.vnc_port = self._vnc_ports[idx]
            inner_env.server_port = self._server_ports[idx]
            inner_env.chromium_port = self._chromium_ports[idx]
            inner_env.vlc_port = self._vlc_ports[idx]
            pool.append(inner_env)
            _cache_dirs.append(_cache_dir)
        self.reserved_env: DesktopEnv = pool.pop()
        self.env_pool = pool
        self.cache_dir_root = cache_dir
        self._env_pool_actions = [[] for _ in range(n_instances_per_task)]
        self._reserved_env_actions: List[Dict[str, any]] = []
        self._reserved_env_step_no = 0

        # episodic stuffs, like counters, will be updated or reset
        self._traj_no: int = -1
        return

    @time_it
    def _save_state(self, snapshot_name=None):
        raise NotImplementedError("Not available for PooledDocker.")

    @time_it
    def close(self):
        # Close all (release) the virtual machine
        for env in self.env_pool:
            try:
                env.close()
            except Exception as e:
                pass
        try:
            self.reserved_env.close()
        except Exception as e:
            pass

        # remove all port locks
        lock_dir = PORT_LOCK_DIR
        for port in self._vnc_ports + self._server_ports + self._chromium_ports + self._vlc_ports:
            lock_fname = os.path.join(lock_dir, f"{port}.lock")
            if os.path.exists(lock_fname):
                # check if pid is us
                with open(lock_fname, "r") as fread:
                    pid = int(fread.read())
                if pid == os.getpid():
                    os.remove(lock_fname)

        # remove all caches
        if os.path.exists(self.cache_dir_root):
            shutil.rmtree(self.cache_dir_root)
        return

    def __execute_no_sleep(self, env: DesktopEnv):
        commands_to_run = [
            ### need DBUS because somehow $UID is empty -> causes error when directly setting gsettings
            "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u user)/bus gsettings set org.gnome.desktop.session idle-delay 0",
            "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u user)/bus gsettings set org.gnome.desktop.screensaver lock-enabled false"
        ]

        no_sleep_success = False
        err = None
        for command in commands_to_run:
            logger.debug(f'Executing no sleep command: {command}')
            payload = json.dumps({"command": command, "shell": True})

            retry_times = 3
            for _ in range(retry_times):
                try:
                    response = requests.post(
                        env.controller.http_server + "/execute",
                        headers={'Content-Type': 'application/json'},
                        data=payload,
                        timeout=15
                    )
                    logger.info(f'no sleep command executed with status {response.json()}')
                    # if first command is successful, we are technically done
                    no_sleep_success = True
                    break
                except Exception as e:
                    err = e
                    time.sleep(3)
                    continue
        
        if not no_sleep_success:
            raise err
        return

    def _post_reset(self, env: DesktopEnv):
        """additional initialization after reset
        """
        env._unpause_emulator()

        ## additional setting to help with the environment
        self.__execute_no_sleep(env)

        env._pause_emulator()
        return

    def _full_reset_single_env(
        self,
        env: DesktopEnv,
        task_config, seed, options
    ):
        _ = env.thread_safe_reset(task_config, seed, options)
        self._post_reset(env)
        return

    def _reset_all_simu_envs(
        self,
        task_config, seed, options,
    ):
        logger.info(f"resetting {len(self.env_pool)} simu envs")
        max_workers = 10
        n_batches = math.ceil(len(self.env_pool) / max_workers)
        wait_time = n_batches * 180  # some setup requires downloading files, which can be slow

        all_completed = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for env in self.env_pool:
                future = executor.submit(
                    self._full_reset_single_env,
                    env,
                    task_config,
                    seed,
                    options
                )
                futures.append(future)
            # timeout by concurrent.futures.as_completed does NOT INTERRUPT if the thread is not terminating
            logger.info(f"Waiting for all simu envs to reset")
            done, not_done = concurrent.futures.wait(
                futures,
                timeout=wait_time,
            )
            logger.info(f"Done: {len(done)}, Not done: {len(not_done)}")

            for future in done:
                future.result()
            
            if len(not_done) > 0:
                all_completed = False
                for future in not_done:
                    future.cancel()  # try to cancel if it has not started running
        
        self._env_pool_actions = [[] for _ in range(len(self.env_pool))]

        # # the simple way to reset all envs
        # for env_idx, env in enumerate(self.env_pool):
        #     logger.info(f"Resetting env {env_idx=}")
        #     env.reset(task_config, seed, options)
        #     self._post_reset(env)
        #     self._env_pool_actions[env_idx] = []

        if not all_completed:
            logger.error(f"Closing all environments in case of resource leaks")
            for env in self.env_pool:
                try:
                    env.close()
                except Exception as e:
                    pass
            raise TimeoutError("Timedout in resetting all simu envs")
        return

    def _reset_reserved_env(
        self, task_config, seed=None, options=None
    ):
        observation = self.reserved_env.reset(task_config, seed, options)
        self._post_reset(self.reserved_env)
        self._reserved_env_actions = []
        return observation

    @time_it
    def reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        # Reset to certain task in OSWorld
        logger.info("Resetting environment...")
        logger.info("Switching task...")
        logger.info("Setting counters...")
        self._traj_no += 1

        self._reset_all_simu_envs(task_config, seed, options)

        # reset the reserved env
        logger.info(f"Resetting reserved env")
        observation = self._reset_reserved_env(task_config, seed, options)
        return observation

    def _get_unused_env_ids(self, **kwargs):
        unused_indices = []
        for idx, action_hist in enumerate(self._env_pool_actions):
            if len(action_hist) == 0:
                unused_indices.append(idx)
        return unused_indices

    @time_it
    def simu_reset(self, env_idx: int, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        """reset using a specific env in the pool
        """
        logger.debug(f"Resetting env {env_idx} for simu_reset")

        selected_env: DesktopEnv = self.env_pool[env_idx]
        observation = selected_env.reset(task_config, seed, options)
        self._env_pool_actions[env_idx] = []
        return observation
    
    @time_it
    def _get_obs(self):
        return self.reserved_env._get_obs()

    @property
    def vm_platform(self):
        return self.reserved_env.vm_platform

    @property
    def vm_screen_size(self):
        return self.reserved_env.vm_screen_size

    @time_it
    def step(self, action, pause=0.5):
        self._reserved_env_step_no += 1
        self._reserved_env_actions.append(action)

        return self.reserved_env.step(action, pause)

    @time_it
    def simu_step(self, action, env_idx: int, pause=0.5):
        # which env to simulate
        selected_env: DesktopEnv = self.env_pool[env_idx]
        _action_hist = self._env_pool_actions[env_idx]
        _action_hist.append(action)

        logger.debug(f'Executing {action} in simu env {env_idx}')
        logger.debug(f"Action history for simu env {env_idx}: {_action_hist}")
        
        return selected_env.step(action, pause)

    @time_it
    def evaluate(self):
        """
        Evaluate whether the task is successfully completed.
        """
        return self.reserved_env.evaluate()

    @time_it
    def simu_evaluate(self, env_idx: int):
        selected_env: DesktopEnv = self.env_pool[env_idx]
        return selected_env.evaluate()

    @time_it
    def render(self, mode='rgb_array'):
        return self.reserved_env.render(mode)


class DynamicPooledDesktopEnv(PooledDesktopEnv):
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
        self.n_instances_per_task = n_instances_per_task
        self._actual_sim_instances = n_instances_per_task  # will grow
        self._env_kwargs = {
            "provider_name": provider_name,
            "region": region,
            "path_to_vm": path_to_vm,
            "snapshot_name": snapshot_name,
            "action_space": action_space,
            "screen_size": screen_size,
            "headless": headless,
            "require_a11y_tree": require_a11y_tree,
            "require_terminal": require_terminal,
            "os_type": os_type
        }
        return

    def _get_unused_env_ids(self, task_config, seed=None, options=None, create_new_if_empty=True):
        unused_indices = super()._get_unused_env_ids()
        if len(unused_indices) == 0 and create_new_if_empty:
            ## create a new env
            logger.info(f"Running out of envs. Creating a new env")
            new_env_idx = self._actual_sim_instances  # starts from zero
            self._actual_sim_instances += 1
            
            more_vnc_ports = _get_n_available_ports(1, self._vnc_ports[-1])
            more_server_ports = _get_n_available_ports(1, self._server_ports[-1])
            more_chromium_ports = _get_n_available_ports(1, self._chromium_ports[-1])
            more_vlc_ports = _get_n_available_ports(1, self._vlc_ports[-1])

            # uses self._actual_sim_instances instead of new_env_idx because reserved_env is not included in the pool
            _cache_dir = os.path.join(self.cache_dir_root, f"env_{self._actual_sim_instances}")
            inner_env = DesktopEnv(
                cache_dir=_cache_dir,
                **self._env_kwargs
            )
            inner_env.vnc_port = more_vnc_ports[0]
            inner_env.server_port = more_server_ports[0]
            inner_env.chromium_port = more_chromium_ports[0]
            inner_env.vlc_port = more_vlc_ports[0]
            self._vnc_ports.append(more_vnc_ports[0])
            self._server_ports.append(more_server_ports[0])
            self._chromium_ports.append(more_chromium_ports[0])
            self._vlc_ports.append(more_vlc_ports[0])

            self.env_pool.append(inner_env)
            self._env_pool_actions.append([])

            ### resetting
            logger.info(f"Resetting newly created env")
            observation = inner_env.reset(task_config, seed, options)
            self._post_reset(inner_env)

            ### return
            logger.info(f"New env created with index {new_env_idx}")
            unused_indices = [new_env_idx]
        return unused_indices

    def _close_simu_env(self, simu_envs: List[DesktopEnv]):
        # Close all (release) the virtual machine
        logger.debug(f"Closing {len(simu_envs)} additionally created simu envs")
        vnc_ports = []
        server_ports = []
        chromium_ports = []
        vlc_ports = []
        for env in simu_envs:
            vnc_ports.append(env.vnc_port)
            server_ports.append(env.server_port)
            chromium_ports.append(env.chromium_port)
            vlc_ports.append(env.vlc_port)
            try:
                env.close()
                self._vnc_ports.remove(env.vnc_port)
                self._server_ports.remove(env.server_port)
                self._chromium_ports.remove(env.chromium_port)
                self._vlc_ports.remove(env.vlc_port)
            except Exception as e:
                pass

        # remove all port locks
        lock_dir = PORT_LOCK_DIR
        for port in vnc_ports + server_ports + chromium_ports:
            lock_fname = os.path.join(lock_dir, f"{port}.lock")
            if os.path.exists(lock_fname):
                # check if pid is us
                with open(lock_fname, "r") as fread:
                    pid = int(fread.read())
                if pid == os.getpid():
                    os.remove(lock_fname)
        return

    @time_it
    def reset(self, task_config: Optional[dict[str, Any]] = None, seed=None, options=None) -> dict[str, Any]:
        # Reset to certain task in OSWorld
        # also reset simu instances back to n_instances_per_task
        logger.info("Resetting environment...")
        logger.info("Switching task...")
        logger.info("Setting counters...")
        self._traj_no += 1

        ### key stats to restore before _reset_all_simu_envs
        # self.env_pool, self._actual_sim_instances
        # self._vnc_ports, self._server_ports, self._chromium_ports
        additional_envs = self.env_pool[self.n_instances_per_task:]
        self._close_simu_env(additional_envs)
        
        self._actual_sim_instances = self.n_instances_per_task
        self.env_pool = self.env_pool[:self.n_instances_per_task]
        self._reset_all_simu_envs(task_config, seed, options)

        # reset the reserved env
        logger.info(f"Resetting reserved env")
        observation = self._reset_reserved_env(task_config, seed, options)
        return observation
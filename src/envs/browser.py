import json
import os
import logging
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union
from tqdm.asyncio import tqdm as atqdm

from playwright.async_api._generated import Browser
from src.envs.utils import atimeout, aretry_timeout
import requests
from beartype import beartype
from gymnasium import Env
from gymnasium.spaces import Box, Text
from playwright.async_api import (
    async_playwright, Playwright,
    CDPSession, Page,
    ViewportSize
)


DATASET = os.environ["DATASET"]
if DATASET == "visualwebarena":
    from browser_env.env_config import (
        CLASSIFIEDS,
        CLASSIFIEDS_RESET_TOKEN,
    )
FORCE_RESET_PER_SETUP_CALL = os.environ.get("FORCE_RESET_PER_SETUP_CALL", "False").lower() == "true"
from browser_env.utils import (
    AccessibilityTree,
    DetachedPage,
    Observation,
)
from browser_env import Trajectory
from src.envs.processors import FastObservationHandler, FastCachedObservationHandler, ObservationMetadata
from src.envs.actions import (
    ActionTypes, Action,
    aexecute_action, get_action_space,
    actionhistory2str, is_equivalent
)
from src.logging import time_it, atime_it


logger = logging.getLogger("logger")


@dataclass
class PlaywrightScript:
    function: str  # goto, get_by_role
    destination: str  # https://www.google.com/, combobox
    name: str | None = None  # Search, Avatar 2009
    operation: str | None = None  # click, fill, press
    value: str | None = None  # avatar movie, Enter


def parse_action(action: str) -> PlaywrightScript:
    splitted = action.strip().split(" ")
    assert len(splitted) >= 2
    match splitted[:2]:
        case ["goto", url]:
            assert len(splitted) == 2
            return PlaywrightScript("goto", url)
        case ["get_by_role", destination]:
            assert len(splitted) >= 4
            match splitted[2:]:
                case [name, operation]:
                    return PlaywrightScript(
                        "get_by_role", destination, name, operation
                    )
                case [name, operation, value]:
                    return PlaywrightScript(
                        "get_by_role", destination, name, operation, value
                    )
                case _:
                    raise ValueError("Invalid action")
        case _:
            raise ValueError(f"Invalid action {action}")


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to stop early"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


class FastBrowserEnv(Env[dict[str, Observation], Action]):
    """
    The goal of this environment is to produce a prototype of a browser environment.
    In the end, we want to support a fully configurable browser environment with wide
    range of action spaces and observation spaces, both structured and unstructured.
    But in this prototype, we just support action space specified by Playwright script,
    and observation space is the html content of the page.
    """

    @beartype
    def __init__(
        self,
        max_page_length: int = 8192,
        headless: bool = True,
        slow_mo: int = 0,
        observation_type: str = "html",
        current_viewport_only: bool = False,
        viewport_size: ViewportSize = {"width": 1280, "height": 720},
        save_trace_enabled: bool = False,
        sleep_after_execution: float = 0.0,
        captioning_fn=None,
    ):
        # TODO: make Space[Action] = ActionSpace
        self.action_space = get_action_space()  # type: ignore[assignment]
        self.headless = headless
        self.slow_mo = slow_mo
        self.current_viewport_only = current_viewport_only
        self.reset_finished = False
        self.viewport_size = viewport_size
        self.save_trace_enabled = save_trace_enabled
        self.sleep_after_execution = sleep_after_execution

        match observation_type:
            case "html" | "accessibility_tree" | "accessibility_tree_with_captioner":
                self.text_observation_type = observation_type
                self.image_observation_type = ""
                self.main_observation_type = "text"
            case "image":
                self.image_observation_type = observation_type
                self.text_observation_type = ""  # type: ignore[assignment]
                self.main_observation_type = "image"
            case "image_som":
                self.image_observation_type = observation_type
                self.text_observation_type = observation_type  # type: ignore[assignment]
                self.main_observation_type = "image"
            case _:
                raise ValueError(
                    f"Unsupported observation type: {observation_type}"
                )

        self.observation_handler = FastObservationHandler(
            self.main_observation_type,
            self.text_observation_type,
            self.image_observation_type,
            self.current_viewport_only,
            self.viewport_size,
            captioning_fn,
        )

        self.observation_space = (
            self.observation_handler.get_observation_space()
        )
        return

    @aretry_timeout(num_retry=6)
    @atimeout(seconds=60)  # maybe environment is resetting
    async def asetup(self, config_file: Path | None = None) -> None:
        if self.reset_finished:
            # exit again in case of retry
            await self.context_manager.__aexit__()
        
        self.context_manager = async_playwright()
        self.playwright: Playwright = await self.context_manager.__aenter__()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless, slow_mo=self.slow_mo
        )

        if config_file:
            with open(config_file, "r") as f:
                instance_config = json.load(f)
        else:
            instance_config = {}

        # Reset site if needed. Currently only supported for Classifieds.
        # TODO(jykoh): Add reset functionality for Shopping/Reddit.
        # this will be handled in an external script
        if instance_config.get("require_reset", False):
            if FORCE_RESET_PER_SETUP_CALL:
                if "classifieds" in instance_config["sites"]:
                    # Send POST request to __CLASSIFIEDS__/index.php?page=reset with token=CLASSIFIEDS_TOKEN
                    response = requests.post(
                        f"{CLASSIFIEDS}/index.php?page=reset",
                        data={"token": CLASSIFIEDS_RESET_TOKEN},
                    )

                    # Check if the request was successful
                    if response.status_code == 200:
                        print("Reset Classifieds site.")
                    else:
                        print(
                            "Failed to reset Classifieds site:",
                            response.status_code,
                        )
                else:
                    print(
                        "WARNING: Reset is not supported for this site. Please manually reset the site."
                    )
            else:
                logger.info(f"skipping reset with {FORCE_RESET_PER_SETUP_CALL=} inside areset. Please handle this in an external script.")

        storage_state = instance_config.get("storage_state", None)
        start_url = instance_config.get("start_url", None)
        geolocation = instance_config.get("geolocation", None)

        # Use custom viewport size if specified in the config, otherwise use the default.
        viewport_size = self.viewport_size.copy()
        viewport_size.update(instance_config.get("viewport_size", {}))
        self.observation_handler.viewport_size = viewport_size

        logger.debug("setting up new_context")
        self.context = await self.browser.new_context(
            viewport=viewport_size,
            storage_state=storage_state,
            geolocation=geolocation,
            device_scale_factor=1,
        )
        if self.save_trace_enabled:
            self.context.tracing.start(screenshots=True, snapshots=True)

        logger.debug("setting up page")
        if start_url:
            start_urls = start_url.split(" |AND| ")
            for url in start_urls:
                logger.debug('starting new page')
                page = await self.context.new_page()
                if self.text_observation_type in [
                    "accessibility_tree",
                    "accessibility_tree_with_captioner",
                ]:
                    logger.debug('starting new_cdp_session')
                    client = await page.context.new_cdp_session(page)
                    logger.debug('sending Accessibility.enable')
                    await client.send("Accessibility.enable")
                    logger.debug("detaching cdp session")
                    await client.detach()
                logger.debug(f"page goto {url=}")
                await page.goto(url)
            logger.debug("bring page to front")
            # set the first page as the current page
            self.page = self.context.pages[0]
            await self.page.bring_to_front()
        else:
            logger.debug("setting up page without start_url")
            self.page = await self.context.new_page()
            if self.text_observation_type in [
                "accessibility_tree",
                "accessibility_tree_with_captioner",
            ]:
                client = await self.page.context.new_cdp_session(self.page)
                await client.send("Accessibility.enable")
                logger.debug("detaching cdp session")
                await client.detach()
        return

    @beartype
    def setup(self, config_file: Path | None = None) -> None:
        raise NotImplementedError("Please use asetup method instead.")

    @atime_it
    async def _aget_obs(self) -> dict[str, Observation]:
        obs = await self.observation_handler.aget_observation(self.page)
        return obs

    def _get_obs(self) -> dict[str, Observation]:
        raise NotImplementedError("Please use aget_obs method instead.")

    @time_it
    def _get_obs_metadata(self) -> dict[str, ObservationMetadata]:
        metadata = self.observation_handler.get_observation_metadata()
        return metadata

    @atime_it
    @beartype
    async def areset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        Reset the environment.
        :param options: options for the environment. The current supported options are:
            - "storage_state": the storage state of the browser. It is a file path to a json file.
        """
        super().reset(seed=seed, options=options)
        if self.reset_finished:
            await self.context_manager.__aexit__()

        if options is not None and "config_file" in options:
            config_file = Path(options["config_file"])
            if config_file.exists():
                await self.asetup(config_file=config_file)
            else:
                raise ValueError(f"Config file {config_file} does not exist.")
        else:
            await self.asetup()
        self.reset_finished = True

        await self.page.wait_for_timeout(int(self.sleep_after_execution * 1000))

        observation = await self._aget_obs()
        observation_metadata = self._get_obs_metadata()
        info = {
            "page": DetachedPage(self.page.url, ""),
            "fail_error": "",
            "observation_metadata": observation_metadata,
        }

        return (observation, info)

    @beartype
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        raise NotImplementedError("Please use areset method instead.")

    async def asave_trace(self, trace_path: str | Path) -> None:
        if self.save_trace_enabled:
            await self.context.tracing.stop(path=trace_path)
        return

    async def aclose(self) -> None:
        if self.reset_finished:
            await self.context_manager.__aexit__()
        return

    def close(self) -> None:
        raise NotImplementedError("Please use aclose method instead.")

    @atime_it
    async def astep(
        self, action: Action
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:
        if not self.reset_finished:
            raise RuntimeError("Call reset first before calling step.")

        success = False
        fail_error = ""
        try:
            self.page = await aexecute_action(
                action,
                self.page,
                self.context,
                self.observation_handler.action_processor,
                self.sleep_after_execution,
            )
            success = True
        except Exception as e:
            fail_error = str(e)
            logger.error(e, exc_info=True)
            logger.info("(Failed at Action):")
            for k_, v_ in action.items():
                logger.info(f"{k_}: {v_}")

        observation = await self._aget_obs()
        observation_metadata = self._get_obs_metadata()

        info = {
            "page": DetachedPage(self.page.url, await self.page.content()),
            "fail_error": fail_error,
            "observation_metadata": observation_metadata,
        }
        msg = (
            observation,
            float(success),  # reward
            False,  # terminated
            False,  # truncated
            info,
        )
        return msg

    def step(
        self, action: Action
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:
        raise NotImplementedError("Please use astep method instead.")


async def hash_page(
    page: Page,
    action_history: list[Action],
    action_set_tag: str
):
    ### need action history since web accessibility tree does not consider scroll down
    act_hist_str = actionhistory2str(action_history, action_set_tag)

    ### also consdier web content since if it changes, cache SHOULD be refreshed
    client = await page.context.new_cdp_session(page)
    _resp = await client.send(
        "Accessibility.getFullAXTree", {}
    )
    accessibility_tree = _resp["nodes"]
    all_nodes = []
    for node in accessibility_tree:
        all_nodes.append((str(node["nodeId"]), str(node["role"]["value"])))
    # sort with node id
    all_nodes.sort(key=lambda x: x[0])
    node_hashed = hashlib.sha256(str(all_nodes).encode('utf-8')).hexdigest()

    ### final hash
    hash_code = f"{act_hist_str=}::{node_hashed=}"
    return hash_code


class FastCachedBrowserEnv(FastBrowserEnv):
    """
    The goal of this environment is to produce a prototype of a browser environment.
    In the end, we want to support a fully configurable browser environment with wide
    range of action spaces and observation spaces, both structured and unstructured.
    But in this prototype, we just support action space specified by Playwright script,
    and observation space is the html content of the page.
    """
    @beartype
    def __init__(
        self,
        max_page_length: int = 8192,
        headless: bool = True,
        slow_mo: int = 0,
        action_set_tag: str = "id_accessibility_tree",
        observation_type: str = "html",
        current_viewport_only: bool = False,
        viewport_size: ViewportSize = {"width": 1280, "height": 720},
        save_trace_enabled: bool = False,
        sleep_after_execution: float = 0.0,
        captioning_fn=None,
    ):
        # TODO: make Space[Action] = ActionSpace
        self.action_space = get_action_space()  # type: ignore[assignment]
        self.headless = headless
        self.slow_mo = slow_mo
        self.action_set_tag = action_set_tag
        self.current_viewport_only = current_viewport_only
        self.reset_finished = False
        self.viewport_size = viewport_size
        self.save_trace_enabled = save_trace_enabled
        self.sleep_after_execution = sleep_after_execution

        match observation_type:
            case "html" | "accessibility_tree" | "accessibility_tree_with_captioner":
                self.text_observation_type = observation_type
                self.image_observation_type = ""
                self.main_observation_type = "text"
            case "image":
                self.image_observation_type = observation_type
                self.text_observation_type = ""  # type: ignore[assignment]
                self.main_observation_type = "image"
            case "image_som":
                self.image_observation_type = observation_type
                self.text_observation_type = observation_type  # type: ignore[assignment]
                self.main_observation_type = "image"
            case _:
                raise ValueError(
                    f"Unsupported observation type: {observation_type}"
                )

        self.observation_handler = FastCachedObservationHandler(
            self.main_observation_type,
            self.text_observation_type,
            self.image_observation_type,
            self.current_viewport_only,
            self.viewport_size,
            captioning_fn,
        )

        self.observation_space = (
            self.observation_handler.get_observation_space()
        )
        return

    @atime_it
    async def astep(
        self, action: Action
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:
        if not self.reset_finished:
            raise RuntimeError("Call reset first before calling step.")

        success = False
        fail_error = ""
        try:
            self.page = await aexecute_action(
                action,
                self.page,
                self.context,
                self.observation_handler.action_processor,
                self.sleep_after_execution,
            )
            success = True
        except Exception as e:
            fail_error = str(e)
            logger.error(e, exc_info=True)
            logger.info("(Failed at Action):")
            logger.info(action.to_simple_str())

        ### added code due to caching
        self.action_history.append(action)
        self.page_hash = await hash_page(
            self.page, self.action_history, self.action_set_tag
        )
        observation = await self._aget_obs()
        observation_metadata = self._get_obs_metadata()

        info = {
            "page": DetachedPage(self.page.url, await self.page.content()),
            "fail_error": fail_error,
            "observation_metadata": observation_metadata,
        }
        msg = (
            observation,
            float(success),  # reward
            False,  # terminated
            False,  # truncated
            info,
        )
        return msg

    @atime_it
    @beartype
    async def areset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        Reset the environment.
        :param options: options for the environment. The current supported options are:
            - "storage_state": the storage state of the browser. It is a file path to a json file.
        """
        super(FastBrowserEnv, self).reset(seed=seed, options=options)
        if self.reset_finished:
            await self.context_manager.__aexit__()

        if options is not None and "config_file" in options:
            config_file = Path(options["config_file"])
            if config_file.exists():
                await self.asetup(config_file=config_file)
            else:
                raise ValueError(f"Config file {config_file} does not exist.")
        else:
            await self.asetup()
        self.reset_finished = True

        await self.page.wait_for_timeout(int(self.sleep_after_execution * 1000))

        ### added code due to caching
        self.action_history = []
        self.page_hash = await hash_page(
            self.page, self.action_history, self.action_set_tag
        )
        observation = await self._aget_obs()
        observation_metadata = self._get_obs_metadata()

        info = {
            "page": DetachedPage(self.page.url, ""),
            "fail_error": "",
            "observation_metadata": observation_metadata,
        }

        return (observation, info)

    async def _aget_obs(self) -> dict[str, Observation]:
        # action_history_str that lead to the current state from areset
        obs = await self.observation_handler.cached_aget_observation(
            self.page,
            cache_id=self.page_hash
        )
        return obs


def is_same_element(element_a: dict, element_b: dict):
    a_text = element_a['text'].lower()
    a_text = a_text[a_text.find(' ') + 1:]  # remove the [xxx] in front
    b_text = element_b['text'].lower()
    b_text = b_text[b_text.find(' ') + 1:]

    a_words = set(a_text.split())
    b_words = set(b_text.split())

    num_similar_words = len(a_words.intersection(b_words))
    if num_similar_words / len(a_words) >= 0.75:
        logger.debug(f"treating element_a={element_a['text']}, element_b={element_b['text']} as the same.")
        return True
    return False


class FastCachedwActionMatchingBrowserEnv(FastCachedBrowserEnv):
    """
    Browser with the ability to:
    + use async to get accessiblity tree
    + use returned cached obs for fast_astep
    + astep uses action matching to patch env non-determinism
    """
    def maybe_update_action_id(self, action: Action, info: dict = None) -> Action:
        if info is not None:
            env_obs_metadata = info['observation_metadata']
            env_obs_text_nodes_info_ = env_obs_metadata['text'].get('obs_nodes_info', {})
            env_obs_text_nodes_info = {k: v['text'] for k, v in env_obs_text_nodes_info_.items()}
            env_obs_som_nodes_info = env_obs_metadata['image'].get('obs_nodes_semantic_info', {})
        else:
            env_obs_metadata = self._curr_obs['info']['observation_metadata']
            env_obs_text_nodes_info_ = env_obs_metadata['text'].get('obs_nodes_info', {})
            env_obs_text_nodes_info = {k: v['text'] for k, v in env_obs_text_nodes_info_.items()}
            env_obs_som_nodes_info = env_obs_metadata['image'].get('obs_nodes_semantic_info', {})

        env_obs_nodes_info = {}
        if len(env_obs_som_nodes_info) > 0:
            env_obs_nodes_info = env_obs_som_nodes_info
        elif len(env_obs_text_nodes_info) > 0:
            env_obs_nodes_info = env_obs_text_nodes_info
        else:
            logger.info(f"maybe_update_action: both text and image has no nodes, skipping")
            return action

        # decide which action obs node it is
        if 'obs_metadata' not in action.metadata:
            logger.debug(f"obs_metadata not found in action={action.to_simple_str()}, skippping")
            return action
        action_obs_nodes_info = {}
        action_obs_text_nodes_info = action.metadata['obs_metadata'].get('text', {}).get('obs_nodes_info', {})
        action_obs_som_nodes_info = action.metadata['obs_metadata'].get('image', {}).get('obs_nodes_semantic_info', {})
        if len(action_obs_som_nodes_info) > 0:
            action_obs_nodes_info = action_obs_som_nodes_info
        elif len(action_obs_text_nodes_info) > 0:
            action_obs_nodes_info = {k: v['text'] for k, v in action_obs_text_nodes_info.items()}

        
        action_element_id = action.element_id
        if action_element_id == '':
            return action
        if action_element_id not in action_obs_nodes_info:
            logger.debug(f"action_element_id={action_element_id} not found in its own nodes={action_obs_nodes_info.keys()}, skipping")
            return action
        
        if action_element_id in env_obs_nodes_info:
            # check if element is matched
            env_node = {
                'text': env_obs_nodes_info[action_element_id]
            }
            action_node = {
                'text': action_obs_nodes_info[action_element_id]
            }
            if is_same_element(env_node, action_node):
                return action
            else:
                logger.info(f"found element might have changed from {action_obs_nodes_info[action_element_id]} to {env_obs_nodes_info[action_element_id]}.")

        logger.info(f'maybe_update_action trying to update action={action.to_simple_str()}')
        logger.info(f'maybe_update_action env_obs_nodes_info={env_obs_nodes_info.keys()}')
        logger.info(f'maybe_update_action action_obs_nodes_info={action_obs_nodes_info.keys()}')
        
        error_margin = int(0.1 * len(action_obs_nodes_info))
        error_margin = max(1, error_margin)
        # assume root node is the min
        action_min_node_id = min([int(k) for k in action_obs_nodes_info.keys()])
        action_element_id_offset = int(action_element_id) - action_min_node_id
        env_min_node_id = min([int(k) for k in env_obs_nodes_info.keys()])

        ## start from middle and search for left and right
        is_updated = False
        for i in range(error_margin+1):
            possible_id = str(action_element_id_offset + env_min_node_id + i)
            if possible_id in env_obs_nodes_info:
                env_node = {
                    'text': env_obs_nodes_info[possible_id]
                }
                action_node = {
                    'text': action_obs_nodes_info[action_element_id]
                }
                if is_same_element(env_node, action_node):
                    # do the substitution
                    previous_raw_prediction = action.raw_prediction
                    action.metadata['previous_raw_prediction'] = previous_raw_prediction
                    action.metadata['previous_element_id'] = action_element_id

                    action.element_id = possible_id
                    action.metadata['obs_metadata'] = env_obs_metadata
                    action.raw_prediction = previous_raw_prediction.replace(f"[{action_element_id}]", f"[{possible_id}]")
                    logger.info(f"maybe_update_action updated action={action.to_simple_str()}")
                    is_updated = True
                    break
            
            possible_id = str(action_element_id_offset + env_min_node_id - i)
            if possible_id in env_obs_nodes_info:
                env_node = {
                    'text': env_obs_nodes_info[possible_id]
                }
                action_node = {
                    'text': action_obs_nodes_info[action_element_id]
                }
                if is_same_element(env_node, action_node):
                    # do the substitution
                    previous_raw_prediction = action.raw_prediction
                    action.metadata['previous_raw_prediction'] = previous_raw_prediction
                    action.metadata['previous_element_id'] = action_element_id

                    action.element_id = possible_id
                    action.metadata['obs_metadata'] = env_obs_metadata
                    action.raw_prediction = previous_raw_prediction.replace(f"[{action_element_id}]", f"[{possible_id}]")
                    logger.info(f"maybe_update_action updated action={action.to_simple_str()}")
                    is_updated = True
                    break
        if not is_updated:
            logger.info(f"maybe_update_action failed to update action.")
        return action

    @beartype
    async def astep(
        self, action: Action
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:
        action = self.maybe_update_action_id(action)

        msg = await super().astep(action)
        obs = msg[0]
        info = msg[-1]
        
        # obs corresponding to self.page
        self._curr_obs = {
            "obs": obs,
            "info": info,
        }
        return msg

    @beartype
    async def areset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        obs, info = await super().areset(seed=seed, options=options)
        self._curr_obs = {
            "obs": obs,
            "info": info,
        }
        return (obs, info)
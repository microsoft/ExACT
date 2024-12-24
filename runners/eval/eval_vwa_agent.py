"""Script to run end-to-end evaluation on the benchmark.

Modified from https://github.com/kohjingyu/search-agents/blob/main/run.py.
"""
import lzma
import pickle
import glob
import json
import logging
import os
import subprocess
import tempfile
import time
import asyncio
import requests
import torch
from PIL import Image
from pathlib import Path
from agent.prompts import *
from browser_env import (
    ActionTypes,
    StateInfo,
    Trajectory,
)
from browser_env.auto_login import get_site_comb_from_filepath
from src.llms.providers.openai_utils import get_all_token_usage, _compute_token_usage_diff
from src.constants import TOKEN_USAGE
from src.evaluation import image_utils
from src.evaluation.vwa_evaluators import evaluator_router
from src.helper_functions import (
    RenderHelper,
    get_action_description,
)
from src.agent.agent_args import ReinforcedAgentArguments
from src.agent.base_agent import FastAgent, PromptAgent
from src.agent.agent_factory import construct_reinforced_agent, is_agent_search_typed
from src.envs.browser import FastCachedwActionMatchingBrowserEnv, early_stop
from src.envs.actions import (
    create_goto_url_action, create_stop_action
)
from src.logging import setup_logger
from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser


os.environ["TOKENIZERS_PARALLELISM"] = "false"
DATASET = os.environ["DATASET"]


logger = logging.getLogger('logger')


@dataclass
class EvalArguments:
    render: bool = field(
        default=False, metadata={"help": "Render the browser"}
    )
    render_screenshot: bool = field(
        default=True, metadata={"help": "Render the screenshot when saving the .html results"}
    )
    slow_mo: int = field(
        default=0,
        metadata={"help": "Slow down the browser by the specified amount"},
    )
    observation_type: str = field(
        default="accessibility_tree",
        metadata={
            "help": "Observation type",
            "choices": [
                "accessibility_tree",
                "accessibility_tree_with_captioner",
                "html",
                "image",
                "image_som",
            ],
        },
    )
    current_viewport_only: bool = field(
        default=False,
        metadata={"help": "Only use the current viewport for the observation"},
    )
    viewport_width: int = field(default=1280)
    viewport_height: int = field(default=2048)
    save_trace_enabled: bool = field(default=False)
    sleep_after_execution: float = field(default=2.5)
    max_steps: int = field(default=5)
    test_config_base_dir: str = field(default="")
    eval_captioning_model_device: str = field(
        default="cuda",
        metadata={
            "choices": ["cpu", "cuda"],
            "help": "Device to run eval captioning model on. By default, runs it on CPU."
        },
    )
    eval_captioning_model: str = field(
        default="Salesforce/blip2-flan-t5-xl",
        metadata={
            "choices": ["Salesforce/blip2-flan-t5-xl"],
            "help": "Captioning backbone for VQA-type evals."
        },
    )
    test_idx: str = field(default=None, metadata={"help": "Idx to test"})
    test_start_idx: int = field(default=0)
    test_end_idx: int = field(default=910)
    result_dir: str = field(default="")

    def __post_init__(self):
        assert self.test_config_base_dir != "", "Test config base dir should be specified."
        assert self.result_dir != "", "Result dir should be specified."
        return


def config() -> tuple[EvalArguments, ReinforcedAgentArguments]:
    parser = HfArgumentParser((EvalArguments, ReinforcedAgentArguments))
    eval_args, agent_args = parser.parse_args_into_dataclasses()

    ## cross checks
    if (
        agent_args.action_set_tag == "id_accessibility_tree"
        and eval_args.observation_type
        not in [
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "image_som",
        ]
    ):
        raise ValueError(
            f"Action type {agent_args.action_set_tag} is incompatible with the observation type {eval_args.observation_type}"
        )
    return eval_args, agent_args


def save_trajectory(task_id: int, trajectory: Trajectory, result_dir: str) -> None:
    traj_root_path = os.path.join(result_dir, "trajectories")
    traj_save_path = os.path.join(traj_root_path, f"task_{task_id}.pkl.xz")
    with lzma.open(traj_save_path, "wb") as fwrite:
        pickle.dump(trajectory, fwrite)
    return


async def aevaluate_single_task(
    eval_args: EvalArguments,
    agent_args: ReinforcedAgentArguments,
    config_file: str,
    eval_caption_image_fn,
    early_stop_thresholds: dict[str, int],
    caption_image_fn,
    agent: FastAgent
):
    env = FastCachedwActionMatchingBrowserEnv(  # used specifically for search type algorithms
        headless=not eval_args.render,
        slow_mo=eval_args.slow_mo,
        action_set_tag=agent_args.action_set_tag,  # used by action caching
        observation_type=eval_args.observation_type,
        current_viewport_only=eval_args.current_viewport_only,
        viewport_size={
            "width": eval_args.viewport_width,
            "height": eval_args.viewport_height,
        },
        save_trace_enabled=eval_args.save_trace_enabled,
        sleep_after_execution=eval_args.sleep_after_execution,
        # NOTE: captioning_fn here is used for LLM + captioning baselines.
        # This can be different from the captioning model used for evals.
        captioning_fn=caption_image_fn,
    )

    render_helper = RenderHelper(
        config_file, eval_args.result_dir, agent_args.action_set_tag
    )

    max_steps = eval_args.max_steps
    try:
        # Load task.
        with open(config_file) as f:
            _c = json.load(f)
            intent = _c["intent"]
            task_id = _c["task_id"]
            image_paths = _c.get("image", None)
            images = []

            # automatically login
            if _c["storage_state"]:
                cookie_file_name = os.path.basename(_c["storage_state"])
                comb = get_site_comb_from_filepath(cookie_file_name)
                temp_dir = tempfile.mkdtemp()
                # subprocess to renew the cookie
                subprocess.run(
                    [
                        "python",
                        "-m",
                        "browser_env.auto_login",
                        "--auth_folder",
                        temp_dir,
                        "--site_list",
                        *comb,
                    ]
                )
                _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                assert os.path.exists(_c["storage_state"])
                # update the config file
                config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                with open(config_file, "w") as f:
                    json.dump(_c, f)

            # Load input images for the task, if any.
            if image_paths is not None:
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                for image_path in image_paths:
                    # Load image either from the web or from a local path.
                    if image_path.startswith("http"):
                        req = requests.get(
                            image_path,
                            headers={"User-Agent": "Mozilla/5.0"},
                            stream=True
                        )
                        input_image = Image.open(req.raw)
                    else:
                        input_image = Image.open(image_path)

                    images.append(input_image)

        task_info = {
            "config_file": config_file,
            "task_id": task_id,
            "intent": intent,
            "images": images,
        }

        logger.info(f"[Config file]: {config_file}")
        logger.info(f"[Intent]: {intent}")

        #### start evaluation
        agent.reset(config_file)
        
        ### NEW hooks for agent to enable memory over time
        agent.on_task_start(
            task_info=task_info,
        )
        trajectory: Trajectory = []
        action_history = []  # Save the action history for the agent so that we can backtrack.
        obs, info = await env.areset(options={"config_file": config_file})
        state_info: StateInfo = {"observation": obs, "info": info, "url": env.page.url}
        trajectory.append(state_info)

        meta_data = {"action_history": ["None"]}
        step_idx = 0
        while True:
            step_idx += 1
            early_stop_flag, stop_info = early_stop(
                trajectory, max_steps, early_stop_thresholds
            )

            is_manual_stop_action = False
            if early_stop_flag:
                action = create_stop_action(f"Early stop: {stop_info}")
                is_manual_stop_action = True
                logger.debug(f"Early stop: {stop_info}")
            else:
                try:
                    other_maybe_useful_inputs = {
                        'task_info': task_info,
                        'action_history': action_history,
                        'step_idx': step_idx,
                        'early_stop_fn': lambda traj: early_stop(traj, max_steps, early_stop_thresholds),
                        'env': env,
                        'eval_args': eval_args,
                        'cmd_args': agent_args,
                    }
                    action = await agent.anext_action(
                        trajectory,
                        intent,
                        meta_data=meta_data,
                        additional_inputs=other_maybe_useful_inputs
                    )
                except ValueError as e:
                    # get the error message
                    is_manual_stop_action = True
                    action = create_stop_action(f"ERROR: {str(e)}")
                    logger.error(e, exc_info=True)
            
            if is_agent_search_typed(agent_args.agent_type) and not is_manual_stop_action:
                all_candidates = action['metadata']["all_candidates"]
                best_actions = action['metadata']["best_actions"]
                best_score = action['metadata']["best_score"]
                logger.debug(f'len(best_actions)={len(best_actions)}')
                logger.debug("best_actions=")
                for a_best in best_actions:
                    logger.debug(a_best.to_simple_str())
                logger.debug(f'{best_score=}', )
            else:
                all_candidates = []
                best_actions = [action]
                best_score = None

            stop_trajectory = False
            if is_agent_search_typed(agent_args.agent_type):
                # Reset environment to the actual current state to prepare for taking the best action.
                obs, info = await env.areset(options={"config_file": config_file})
                prev_url = env.page.url
                state_info = {"observation": obs, "info": info, "url": env.page.url}
                
                #### abbreviate action history if the URL has changed
                truncated_action_history = []
                for a_hist in action_history:
                    obs, _, _, _, info = await env.astep(a_hist)
                    curr_url = env.page.url
                    state_info = {"observation": obs, "info": info, "url": env.page.url}

                    # Optimization to simplify the action history, since we will commit the best action.
                    truncated_action_history.append(a_hist)
                    if curr_url != prev_url:
                        # URL has changed, update the truncated_action_history
                        truncated_action_history = [create_goto_url_action(curr_url)]
                        prev_url = curr_url
                action_history = truncated_action_history

            prev_url = env.page.url
            # Now we can actually execute the best action.
            for best_idx, action in enumerate(best_actions):
                #### render the action+curr web environment
                # correct action before rendering
                action = env.maybe_update_action_id(action)
                trajectory.append(action)
                all_candidates.append(f"Selected action {best_idx}: {action['raw_prediction']}")
                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=agent_args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor if isinstance(agent, PromptAgent) else None
                )
                render_helper.render(
                    action,
                    state_info,
                    meta_data,
                    eval_args.render_screenshot,
                    all_candidates if is_agent_search_typed(agent_args.agent_type) else None
                )
                meta_data["action_history"].append(action_str)
                
                #### debugging
                obs_text = state_info["observation"]["text"]
                obs_nodes = state_info["info"]["observation_metadata"]["text"].get("obs_nodes_info", {})
                obs_nodes_som = state_info["info"]["observation_metadata"]["image"].get("obs_nodes_semantic_info", {})
                action_element_id = action['element_id']
                if action_element_id != "":
                    logger.debug("============")
                    if action_element_id in obs_text:
                        logger.debug(f"ACTUAL env.astep: [{action_element_id}] is found on the page!")
                    else:
                        logger.debug(f"ACTUAL env.astep: [{action_element_id}] is NOT found on the page!")
                    if action_element_id in obs_nodes or action_element_id in obs_nodes_som:
                        logger.debug(f"ACTUAL env.astep: [{action_element_id}] is also found in obs_nodes/semantic_info!")
                    else:
                        logger.debug(f"ACTUAL env.astep: [{action_element_id}] is NOT found in obs_nodes/semantic_info!!")
                    logger.debug(f"ACTUAL env.astep: raw_prediction: {action['raw_prediction']}")
                    logger.debug(f"ACTUAL env.astep: obs_text:\n{obs_text}")
                    logger.debug(f"ACTUAL env.astep: obs_nodes:\n{obs_nodes.keys()}")
                    logger.debug(f"ACTUAL env.astep: obs_nodes_som:\n{obs_nodes_som.keys()}")
                    logger.debug("============")

                if action["action_type"] == ActionTypes.STOP:
                    stop_trajectory = True
                    break

                ### Execute the action.
                obs, success, terminated, _, info = await env.astep(action)
                if not success:
                    logger.info(f"[Action Failed]: {info['fail_error']}")

                ### Save the committed action to the action history.
                action_history.append(action)
                curr_url = env.page.url
                if curr_url != prev_url:
                    # URL has changed, simplify the action_history so that we resume from this checkpoint
                    action_history = [create_goto_url_action(curr_url)]
                    prev_url = curr_url
                state_info = {"observation": obs, "info": info, "url": env.page.url}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    stop_trajectory = True
                    break

            # We solved the task and can quit.
            if stop_trajectory or (best_score is not None and best_score == 1.0):
                # Save obs
                break
        # END SEARCH

        # NOTE: eval_caption_image_fn is used for running eval_vqa functions.
        evaluator = evaluator_router(
            config_file, captioning_fn=eval_caption_image_fn
        )
        score = await evaluator(
            trajectory=trajectory,
            config_file=config_file,
            page=env.page
        )

        ### NEW hooks for agent to enable memory over time
        agent.on_task_end(
            trajectory=trajectory,
            score=score,
            task_info=task_info,
            meta_data=meta_data,
        )

        if score == 1:
            logger.info(f"[Result] (PASS) {config_file}")
        else:
            logger.info(f"[Result] (FAIL) {config_file}")

        ## save trajectory
        save_trajectory(task_id, trajectory, eval_args.result_dir)

        if eval_args.save_trace_enabled:
            await env.asave_trace(
                Path(eval_args.result_dir) / "traces" / f"{task_id}.zip"
            )
    except Exception as e:
        logger.info(f"[Unhandled Error] {repr(e)}]")
        import traceback

        # write to error file
        error_text = traceback.format_exc()
        with open(Path(eval_args.result_dir) / "error.txt", "a") as f:
            f.write(f"[Config file]: {config_file}\n")
            f.write(f"[Unhandled Error] {repr(e)}\n")
            f.write(error_text)  # write stack trace to file

        error_rerun_dir = os.path.join(eval_args.result_dir, "error_tids")
        Path(error_rerun_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(error_rerun_dir) / "error_tid.txt", "a") as fwrite:
            fwrite.write(f"{task_id},")
        # also write as a file for parallel runner to rerun
        with open(Path(error_rerun_dir) / f"{task_id}.txt", "a") as fwrite:
            fwrite.write(f"[Unhandled Error] {repr(e)}\n")
            fwrite.write(error_text)
        score = 0.0

    render_helper.close()
    await env.aclose()
    return score


def test(
    eval_args: EvalArguments,
    agent_args: ReinforcedAgentArguments,
    config_file_list: list[str]
) -> None:
    scores = []
    times = []
    _prev_llm_token_usage = {}
    llm_tokens = []

    early_stop_thresholds = {
        "parsing_failure": agent_args.parsing_failure_th,
        "repeating_action": agent_args.repeating_action_failure_th,
    }

    if eval_args.observation_type in [
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, agent_args.captioning_model
        )
    else:
        caption_image_fn = None

    # Load a (possibly different) captioning model for running VQA evals.
    # if DATASET == 'visualwebarena':
    #     if (
    #         caption_image_fn
    #         and eval_args.eval_captioning_model == agent_args.captioning_model
    #     ):
    #         eval_caption_image_fn = caption_image_fn
    #     else:
    #         caption_model_dtype = torch.float32
    #         if torch.cuda.is_available() and eval_args.eval_captioning_model_device == "cuda":
    #             caption_model_dtype = torch.float16
    #         eval_caption_image_fn = image_utils.get_captioning_fn(
    #             eval_args.eval_captioning_model_device,
    #             caption_model_dtype,
    #             eval_args.eval_captioning_model,
    #         )
    # else:
    #     caption_image_fn = None
    #     eval_caption_image_fn = None
    if (
        caption_image_fn
        and eval_args.eval_captioning_model == agent_args.captioning_model
    ):
        eval_caption_image_fn = caption_image_fn
    else:
        caption_model_dtype = torch.float32
        if torch.cuda.is_available() and eval_args.eval_captioning_model_device == "cuda":
            caption_model_dtype = torch.float16
        eval_caption_image_fn = image_utils.get_captioning_fn(
            eval_args.eval_captioning_model_device,
            caption_model_dtype,
            eval_args.eval_captioning_model,
        )

    agent = construct_reinforced_agent(
        agent_args,
        captioning_fn=caption_image_fn if eval_args.observation_type == "accessibility_tree_with_captioner" else None
    )  # NOTE: captioning_fn here is used for captioning input images.

    for config_file in config_file_list:
        start_time = time.time()
        task_score = asyncio.run(aevaluate_single_task(
            eval_args, agent_args,
            config_file,
            eval_caption_image_fn=eval_caption_image_fn,
            early_stop_thresholds=early_stop_thresholds,
            caption_image_fn=caption_image_fn,
            agent=agent
        ))
        time_spent = time.time() - start_time
        logger.info(f"Task {config_file} took {time_spent} seconds")

        scores.append(task_score)
        times.append(time_spent / 60.0)
        all_token_usages_so_far = get_all_token_usage(TOKEN_USAGE)
        # compute diff to get current token usage
        if len(_prev_llm_token_usage) != 0:
            # compute diff
            all_token_usages_so_far = _compute_token_usage_diff(
                prev_all_token_usage=_prev_llm_token_usage,
                curr_all_token_usage=all_token_usages_so_far
            )
        # accumulate token usage
        _prev_llm_token_usage = all_token_usages_so_far
        llm_tokens.append(all_token_usages_so_far)
        logger.info(f"Task {config_file} used {all_token_usages_so_far} tokens")
    
    if len(scores):
        logger.info(f"Average score: {sum(scores) / len(scores)}")
        logger.info(f"Average time: {sum(times) / len(times)} (minutes)")

        ### save performance
        performance_dir = os.path.join(eval_args.result_dir, "performances")
        # save scores separately in case we need to rerun the evaluation
        for r_i, ran_config_file in enumerate(config_file_list):
            tid = int(os.path.basename(ran_config_file).split(".")[0])
            perf_file_name = f"performance_{tid}.json"
            with open(os.path.join(performance_dir, perf_file_name), "w") as fwrite:
                perf = {
                    "eval_configs": config_file_list[r_i],
                    "scores": scores[r_i],
                    "times": times[r_i],
                    "llm_tokens": llm_tokens[r_i],
                }
                json.dump(perf, fwrite, indent=4, sort_keys=True)
    return


def prepare(args: EvalArguments) -> None:
    # convert prompt python files to json/update prompt files if changes are made
    from src.prompts.vwa import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True, exist_ok=True)
    if not (Path(result_dir) / "performances").exists():
        (Path(result_dir) / "performances").mkdir(parents=True, exist_ok=True)
    if not (Path(result_dir) / "trajectories").exists():
        (Path(result_dir) / "trajectories").mkdir(parents=True, exist_ok=True)
    return


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(combined_args: dict, result_dir: str) -> None:
    config_file = Path(result_dir) / "config.json"
    with open(config_file, "w") as f:
        json.dump(combined_args, f, indent=4)
        logger.info(f"Dump config to {config_file}")
    return


if __name__ == "__main__":
    eval_args, agent_args = config()
    prepare(eval_args)

    #### skip tests if already done
    test_config_base_dir = eval_args.test_config_base_dir
    test_file_list = []
    if eval_args.test_idx is not None:
        print(f"Testing on {eval_args.test_idx}")
        for x in eval_args.test_idx.split(","):
            test_file_list.append(os.path.join(test_config_base_dir, f"{x}.json"))
    else:
        print(f"Testing on {eval_args.test_start_idx} to {eval_args.test_end_idx}")
        st_idx = eval_args.test_start_idx
        ed_idx = eval_args.test_end_idx
        for i in range(st_idx, ed_idx):
            test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))
    test_file_list = get_unfinished(test_file_list, eval_args.result_dir)
    print(f"Total {len(test_file_list)} tasks left")
    print(f"{eval_args.render=}")

    if len(test_file_list) == 0:
        print("All tasks are finished.")
        exit(0)

    #### setup logger
    setup_logger(os.path.join(eval_args.result_dir, "log_files"))
    if os.environ.get("DEBUG", "") != "":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    ### default settings
    eval_args.render_screenshot = True
    eval_args.current_viewport_only = True
    dump_config(
        combined_args = {
            "eval_args": asdict(eval_args),
            "agent_args": asdict(agent_args)
        },
        result_dir = eval_args.result_dir
    )

    #### run tests
    test(eval_args, agent_args, test_file_list)

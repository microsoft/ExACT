import datetime
import json
import logging
import os
import time
import lzma
import pickle
from wrapt_timeout_decorator import *
from exact.llms.providers.openai_utils import get_all_token_usage, reset_token_usage
from exact.env.desktop_env import DesktopEnv
from exact.env.desktop_env_dev import PooledDesktopEnv
from exact.env.desktop_env_resettable import ResettableDesktopEnv
from exact.agent.agent_factory import construct_search_metadata
from exact.agent.base import BaseAgent, RAgentMixin, ResumableAgentMixin
from exact.env.desktop_env_utils import (
    ObsPostProcessor, encode_image, resize_image_from_bytes
)
from exact.args import CommonArgs, EnvArgs, AgentArgs


logger = logging.getLogger("src.run_utils")


def compact(d, indent=0):
    def tight(obj):
        return json.dumps(obj, separators=(',', ':'))
    
    out_str = ''
    for i, (k, v) in enumerate(d.items()):
        comma = ',' if i < len(d) else ''
        out_str += f'{" " * indent}{tight(k)}:{tight(v)}{comma}\n'
    return out_str


def _process_raw_action_for_html(raw_response_str: str):
    special_word_replacement = {
        "<think>": "&lt;think&gt;",
        "</think>": "&lt;/think&gt;",
        "<action>": "&lt;action&gt;",
        "</action>": "&lt;/action&gt;",
    }
    for k, v in special_word_replacement.items():
        raw_response_str = raw_response_str.replace(k, v)
    return raw_response_str


def render_trajectory_to_html(task_config: dict, trajectory: list, postprocesser: ObsPostProcessor, output_fpath: str):
    instruction = task_config["instruction"]
    eval_config = task_config["evaluator"]
    eval_config_str = compact(eval_config, indent=4)

    content = f"<pre><em>Instruction:</em> {instruction}</pre>"
    content += f"<pre><em>Evaluator:</em><br/>{eval_config_str}</pre>"
    for data in trajectory:
        if "obs" in data.keys():
            # is observation
            obs = data["obs"]
            processed_obs = postprocesser(obs)
            if postprocesser.observation_type in ["screenshot", "screenshot_a11y_tree"]:
                screenshot = obs['screenshot']
            else:
                screenshot = processed_obs['screenshot'] or obs['screenshot']
            ally_tree = processed_obs['accessibility_tree']

            screenshot = resize_image_from_bytes(screenshot, size=(960, 540))
            screenshot_b64 = encode_image(screenshot)

            content += (
                '<div class="obs">'
                    "<h4>Observation:</h4>"
                    f'<img src="data:image/png;base64,{screenshot_b64}"/>'
                    f'<pre>{ally_tree}</pre>'
                '</div>'
            )
        else:
            # is action
            raw_action = _process_raw_action_for_html(data["raw_action"])
            content += (
                '<div class="raw_action">'
                    '<h4>Raw Action:</h4>'
                    f'<pre>{raw_action}</pre>'
                '</div>'
            )
            content += (
                '<div class="action">'
                    f'<pre>{data["action"]}</pre>'
                '</div>'
            )
    
    style = (
        ".raw_action {background-color: grey;}\n"
        ".action {background-color: yellow;}\n"
        "pre {white-space: pre-wrap; word-wrap: break-word;}"
    )
    HTML_TEMPLATE = (
        "<html>\n"
        "<head>\n"
            "<style>\n"
                f"{style}\n"
            "</style>\n"
        "</head>\n"
            "<body>\n"
                f"{content}\n"
            "</body>\n"
        "</html>\n"
    )
    with open(output_fpath, "w") as fwrite:
        fwrite.write(HTML_TEMPLATE)
    return


def save_full_trajectory(trajectory, output_fpath):
    # in case we put nasty classes into the trajectory
    with lzma.open(output_fpath, "wb") as fwrite:
        pickle.dump(trajectory, fwrite)
    return


def run_single_example(
    agent: BaseAgent,
    env: DesktopEnv,
    task_config: dict,
    common_args: CommonArgs,
    env_args: EnvArgs,
    agent_args: AgentArgs,
    result_dir: str
):
    max_steps = common_args.max_steps
    instruction = task_config["instruction"]
    ##### init env, agent, and etc.
    reset_token_usage()
    agent.reset()
    obs = env.reset(task_config=task_config)
    done = False
    step_idx = 0
    if env_args.save_recording:
        env.controller.start_recording()

    full_trajectory = [
        {"obs": obs, "info": None, "reward": 0.0, "done": False},
    ]
    html_fpath = os.path.join(result_dir, "trajectory.html")
    render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)

    ##### action loop
    start_time = time.time()
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, env_args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)

            # update trajectory
            full_trajectory.append({"raw_action": response, "action": action})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})
            render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)
            
            # Save screenshot and trajectory information
            screenshot_fpath = os.path.join(result_dir, f"step_{step_idx + 1}_{action_timestamp}.png")
            with open(screenshot_fpath,"wb") as _fwrite:
                _fwrite.write(obs['screenshot'])
            with open(os.path.join(result_dir, "traj.jsonl"), "a") as f:
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
                logger.info("The episode is done.")
                break
        step_idx += 1

    time_spent = time.time() - start_time
    logger.info(f"Time spent: {time_spent/60.0:.2f}min")

    ### first we save the trajectory
    traj_fpath = os.path.join(result_dir, "trajectory.pkl.xz")
    save_full_trajectory(full_trajectory, traj_fpath)

    ### compute score and token usage
    result = env.evaluate()
    result = float(result)
    logger.info(f"Result: {result:.2f}")

    all_token_usages_so_far = get_all_token_usage()
    logger.info(f"Token consumption: {all_token_usages_so_far})")

    with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as fwrite:
        fwrite.write(f"{result}\n")
    with open(os.path.join(result_dir, "performance.json"), "w") as fwrite:
        perf = {
            "score": result,
            "llm_token": all_token_usages_so_far,
            "time_spent": time_spent/60.0,
        }
        json.dump(perf, indent=4, sort_keys=True, fp=fwrite)
    
    ### end recording
    if env_args.save_recording:
        env.controller.end_recording(os.path.join(result_dir, "recording.mp4"))
    return {
        'score': result,
        'llm_token': all_token_usages_so_far,
        'time_spent': time_spent/60.0,
    }


def run_single(
    agent: BaseAgent,
    env: DesktopEnv,
    task_config: dict,
    common_args: CommonArgs,
    env_args: EnvArgs,
    agent_args: AgentArgs,
    result_dir: str
):
    instruction = task_config["instruction"]
    ##### init env, agent, and etc.
    reset_token_usage()
    agent.reset()
    obs = env.reset(task_config=task_config)
    done = False
    step_idx = 0
    if env_args.save_recording:
        env.controller.start_recording()

    full_trajectory = [
        {"obs": obs, "info": None, "reward": 0.0, "done": False},
    ]
    html_fpath = os.path.join(result_dir, "trajectory.html")
    render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)

    ##### action loop
    start_time = time.time()
    while not done and step_idx < common_args.max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(
                action,
                env_args.sleep_after_execution
            )

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)

            # update trajectory
            full_trajectory.append({"raw_action": response, "action": action})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})
            render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)
            
            # Save screenshot and trajectory information
            screenshot_fpath = os.path.join(result_dir, f"step_{step_idx + 1}_{action_timestamp}.png")
            with open(screenshot_fpath,"wb") as _fwrite:
                _fwrite.write(obs['screenshot'])
            with open(os.path.join(result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "raw_action": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    time_spent = time.time() - start_time
    logger.info(f"Time spent: {time_spent/60.0:.2f}min")

    ### first we save the trajectory
    traj_fpath = os.path.join(result_dir, "trajectory.pkl.xz")
    save_full_trajectory(full_trajectory, traj_fpath)

    ### compute score and token usage
    result = env.evaluate()
    result = float(result)
    logger.info(f"Result: {result:.2f}")

    all_token_usages_so_far = get_all_token_usage()
    logger.info(f"Token consumption: {all_token_usages_so_far})")

    with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as fwrite:
        fwrite.write(f"{result}\n")
    with open(os.path.join(result_dir, "performance.json"), "w") as fwrite:
        perf = {
            "score": result,
            "llm_token": all_token_usages_so_far,
            "time_spent": time_spent/60.0,
        }
        json.dump(perf, indent=4, sort_keys=True, fp=fwrite)
    
    ### end recording
    if env_args.save_recording:
        env.controller.end_recording(os.path.join(result_dir, "recording.mp4"))
    return {
        'score': result,
        'llm_token': all_token_usages_so_far,
        'time_spent': time_spent/60.0,
    }
    
    
def run_single_resettable(
    agent: BaseAgent,
    env: ResettableDesktopEnv,
    task_config: dict,
    common_args: CommonArgs,
    env_args: EnvArgs,
    agent_args: AgentArgs,
    result_dir: str
):
    instruction = task_config["instruction"]
    ##### init env, agent, and etc.
    reset_token_usage()
    agent.reset()
    obs = env.reset(task_config=task_config)
    done = False
    step_idx = 0
    if env_args.save_recording:
        env.controller.start_recording()

    full_trajectory = [
        {"obs": obs, "info": None, "reward": 0.0, "done": False},
    ]
    html_fpath = os.path.join(result_dir, "trajectory.html")
    render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)

    ##### action loop
    start_time = time.time()
    while not done and step_idx < common_args.max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )
        if len(actions) == 0:
            full_trajectory.append({"raw_action": response, "action": "None", "step_idx": step_idx})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})  # this will be same as last state
        
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(
                action,
                resp_str=response,
                step_idx=step_idx,
                pause=env_args.sleep_after_execution
            )

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)

            # update trajectory
            full_trajectory.append({"raw_action": response, "action": action, "step_idx": step_idx})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})
            render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)
            
            # Save screenshot and trajectory information
            screenshot_fpath = os.path.join(result_dir, f"step_{step_idx + 1}_{action_timestamp}.png")
            with open(screenshot_fpath,"wb") as _fwrite:
                _fwrite.write(obs['screenshot'])
            with open(os.path.join(result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "raw_action": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    time_spent = time.time() - start_time
    logger.info(f"Time spent: {time_spent/60.0:.2f}min")

    ### first we save the trajectory
    traj_fpath = os.path.join(result_dir, "trajectory.pkl.xz")
    save_full_trajectory(full_trajectory, traj_fpath)

    ### compute score and token usage
    result = env.evaluate()
    result = float(result)
    logger.info(f"Result: {result:.2f}")

    all_token_usages_so_far = get_all_token_usage()
    logger.info(f"Token consumption: {all_token_usages_so_far})")

    with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as fwrite:
        fwrite.write(f"{result}\n")
    with open(os.path.join(result_dir, "performance.json"), "w") as fwrite:
        perf = {
            "score": result,
            "llm_token": all_token_usages_so_far,
            "time_spent": time_spent/60.0,
        }
        json.dump(perf, indent=4, sort_keys=True, fp=fwrite)
    
    ### end recording
    if env_args.save_recording:
        env.controller.end_recording(os.path.join(result_dir, "recording.mp4"))
    return {
        'score': result,
        'llm_token': all_token_usages_so_far,
        'time_spent': time_spent/60.0,
    }


def _eval_again(
    env: PooledDesktopEnv,
    env_args: EnvArgs,
    task_config,
    result_dir: str,
    agent: BaseAgent,
    all_response,
    all_actions,
):
    ### replay twice
    # sometimes there is randomness in the environment, causing replay to fail
    obs = env._reset_reserved_env(task_config=task_config)
    done = False
    info = None
    reward = 0.0

    full_trajectory = [
        {"obs": obs, "info": info, "reward": reward, "done": done},
    ]
    html_fpath = os.path.join(result_dir, "trajectory.html")
    with open(html_fpath, "w") as fwrite:
        # overwrite the file
        pass
    render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)

    ##### action loop
    step_idx = 0
    for response, actions in zip(all_response, all_actions):
        if len(actions) == 0:
            full_trajectory.append({"raw_action": response, "action": "None", "step_idx": step_idx})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})  # this will be same as last state
        
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(
                action,
                env_args.sleep_after_execution
            )

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)

            # update trajectory
            full_trajectory.append({"raw_action": response, "action": action, "step_idx": step_idx})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})
            render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)
            
            # Save screenshot and trajectory information
            screenshot_fpath = os.path.join(result_dir, f"step_{step_idx + 1}_{action_timestamp}.png")
            with open(screenshot_fpath,"wb") as _fwrite:
                _fwrite.write(obs['screenshot'])
            with open(os.path.join(result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "raw_action": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    logger.info(f"Replay 2 finished.")

    ### first we save the trajectory
    traj_fpath = os.path.join(result_dir, "trajectory.pkl.xz")
    save_full_trajectory(full_trajectory, traj_fpath)

    ### compute score and token usage
    result = env.evaluate()
    result = float(result)
    logger.info(f"Replay 2 result: {result:.2f}")
    return result


def run_single_search(
    agent: BaseAgent,
    env: PooledDesktopEnv,
    task_config: dict,
    common_args: CommonArgs,
    env_args: EnvArgs,
    agent_args: AgentArgs,
    result_dir: str
):
    instruction = task_config["instruction"]
    ##### init env, agent, and etc.
    reset_token_usage()
    agent.reset()

    logger.info(f"Running experiment with {env.cache_dir_root=}")
    obs = env.reset(task_config=task_config)
    done = False
    info = None
    reward = 0.0
    if env_args.save_recording:
        env.controller.start_recording()

    full_trajectory = [
        {"obs": obs, "info": info, "reward": reward, "done": done},
    ]
    html_fpath = os.path.join(result_dir, "trajectory.html")
    render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)

    ##### action loop
    start_time = time.time()
    search_metadata = construct_search_metadata(
        agent_name=agent.name,
        result_dir=result_dir,
        env=env,
        env_args=env_args,
        task_config=task_config,
    )
    # unlike normal agent, search agent directly output the final TRAJECTORY of actions
    step_idx = 0
    all_response, all_actions = agent.predict(
        instruction,
        obs,
        search_metadata=search_metadata
    )
    for response, actions in zip(all_response, all_actions):
        if len(actions) == 0:
            full_trajectory.append({"raw_action": response, "action": "None", "step_idx": step_idx})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})  # this will be same as last state
        
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(
                action,
                env_args.sleep_after_execution
            )

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)

            # update trajectory
            full_trajectory.append({"raw_action": response, "action": action, "step_idx": step_idx})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})
            render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)
            
            # Save screenshot and trajectory information
            screenshot_fpath = os.path.join(result_dir, f"step_{step_idx + 1}_{action_timestamp}.png")
            with open(screenshot_fpath,"wb") as _fwrite:
                _fwrite.write(obs['screenshot'])
            with open(os.path.join(result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "raw_action": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    time_spent = time.time() - start_time
    logger.info(f"Time spent: {time_spent/60.0:.2f}min")

    ### first we save the trajectory
    traj_fpath = os.path.join(result_dir, "trajectory.pkl.xz")
    save_full_trajectory(full_trajectory, traj_fpath)

    ### compute score and token usage
    result = env.evaluate()
    result = float(result)
    logger.info(f"Result: {result:.2f}")

    # replaying actions again, as sometimes environment have randomness
    if result != 1.0:
        for _ in range(2):
            logger.info(f"Replaying found actions again...")
            result = _eval_again(
                env,
                env_args,
                task_config,
                result_dir,
                agent,
                all_response,
                all_actions
            )
            if result == 1.0:
                break
        logger.info(f"Replaying finished. Result: {result:.2f}")

    all_token_usages_so_far = get_all_token_usage()
    logger.info(f"Token consumption: {all_token_usages_so_far})")

    with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as fwrite:
        fwrite.write(f"{result}\n")
    with open(os.path.join(result_dir, "performance.json"), "w") as fwrite:
        perf = {
            "score": result,
            "llm_token": all_token_usages_so_far,
            "time_spent": time_spent/60.0,
        }
        json.dump(perf, indent=4, sort_keys=True, fp=fwrite)
    
    ### end recording
    if env_args.save_recording:
        env.controller.end_recording(os.path.join(result_dir, "recording.mp4"))
    return {
        'score': result,
        'llm_token': all_token_usages_so_far,
        'time_spent': time_spent/60.0,
    }


def run_single_reinforced_search(
    agent: BaseAgent | RAgentMixin,
    env: PooledDesktopEnv,
    task_config: dict,
    common_args: CommonArgs,
    env_args: EnvArgs,
    agent_args: AgentArgs,
    result_dir: str
):
    instruction = task_config["instruction"]
    ##### init env, agent, and etc.
    reset_token_usage()
    agent.reset()

    # ragent hook
    agent.on_task_start(task_config)

    logger.info(f"Running experiment with {env.cache_dir_root=}")
    obs = env.reset(task_config=task_config)
    done = False
    info = None
    reward = 0.0
    
    if env_args.save_recording:
        env.controller.start_recording()
    
    full_trajectory = [
        {"obs": obs, "info": info, "reward": reward, "done": done},
    ]
    html_fpath = os.path.join(result_dir, "trajectory.html")
    render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)

    ##### action loop
    start_time = time.time()
    search_metadata = construct_search_metadata(
        agent_name=agent.name,
        result_dir=result_dir,
        env=env,
        env_args=env_args,
        task_config=task_config,
    )
    # unlike normal agent, search agent directly output the final TRAJECTORY of actions
    step_idx = 0
    all_response, all_actions = agent.predict(
        instruction,
        obs,
        search_metadata=search_metadata
    )
    for response, actions in zip(all_response, all_actions):
        if len(actions) == 0:
            full_trajectory.append({"raw_action": response, "action": "None", "step_idx": step_idx})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})  # this will be same as last state
        
        ### this loop won't run anyway if no actions
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(
                action,
                env_args.sleep_after_execution
            )

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)

            # update trajectory
            full_trajectory.append({"raw_action": response, "action": action, "step_idx": step_idx})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})
            render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)
            
            # Save screenshot and trajectory information
            screenshot_fpath = os.path.join(result_dir, f"step_{step_idx + 1}_{action_timestamp}.png")
            with open(screenshot_fpath,"wb") as _fwrite:
                _fwrite.write(obs['screenshot'])
            with open(os.path.join(result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "raw_action": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    time_spent = time.time() - start_time
    logger.info(f"Time spent: {time_spent/60.0:.2f}min")

    ### first we save the trajectory
    traj_fpath = os.path.join(result_dir, "trajectory.pkl.xz")
    save_full_trajectory(full_trajectory, traj_fpath)

    ### compute score and token usage
    result = env.evaluate()
    result = float(result)
    logger.info(f"Result: {result:.2f}")

    # replaying actions again, as sometimes environment have randomness
    if result != 1.0:
        for _ in range(2):
            logger.info(f"Replaying found actions again...")
            result = _eval_again(
                env,
                env_args,
                task_config,
                result_dir,
                agent,
                all_response,
                all_actions
            )
            if result == 1.0:
                break
        logger.info(f"Replaying finished. Result: {result:.2f}")

    ### agent on task end
    agent.on_task_end(full_trajectory, task_config, {'success': 1.0 if result == 1.0 else 0.0})

    all_token_usages_so_far = get_all_token_usage()
    logger.info(f"Token consumption: {all_token_usages_so_far})")

    with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as fwrite:
        fwrite.write(f"{result}\n")
    with open(os.path.join(result_dir, "performance.json"), "w") as fwrite:
        perf = {
            "score": result,
            "llm_token": all_token_usages_so_far,
            "time_spent": time_spent/60.0,
        }
        json.dump(perf, indent=4, sort_keys=True, fp=fwrite)
    
    ### end recording
    if env_args.save_recording:
        env.controller.end_recording(os.path.join(result_dir, "recording.mp4"))
    return {
        'score': result,
        'llm_token': all_token_usages_so_far,
        'time_spent': time_spent/60.0,
    }


def _try_loading_agent_state_if_exists(agent: ResumableAgentMixin, agent_fpath: str):
    agent_loaded_successfully = False
    if os.path.exists(agent_fpath):
        logger.info(f"Loading agent from {agent_fpath}")
        try:
            agent.load_state(agent_fpath)
            agent_loaded_successfully = True
        except Exception as e:
            logger.error(f"Failed to load agent state: {e}. Removing and restarting.")
            os.remove(agent_fpath)
    return agent_loaded_successfully


def run_single_resumable_search(
    agent: BaseAgent | ResumableAgentMixin,
    env: PooledDesktopEnv,
    task_config: dict,
    common_args: CommonArgs,
    env_args: EnvArgs,
    agent_args: AgentArgs,
    result_dir: str
):
    instruction = task_config["instruction"]
    ##### init env, agent, and etc.
    reset_token_usage()
    agent.reset()

    logger.info(f"Running experiment with {env.cache_dir_root=}")
    obs = env.reset(task_config=task_config)
    done = False
    info = None
    reward = 0.0
    
    if env_args.save_recording:
        env.controller.start_recording()
    
    full_trajectory = [
        {"obs": obs, "info": info, "reward": reward, "done": done},
    ]
    html_fpath = os.path.join(result_dir, "trajectory.html")
    render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)

    ##### action loop
    start_time = time.time()
    search_metadata = construct_search_metadata(
        agent_name=agent.name,
        result_dir=result_dir,
        env=env,
        env_args=env_args,
        task_config=task_config,
    )
    # unlike normal agent, search agent directly output the final TRAJECTORY of actions
    step_idx = 0
    agent_fpath = os.path.join(result_dir, "agent.state")
    agent_loaded_successfully = _try_loading_agent_state_if_exists(agent, agent_fpath)
    # load agent state and continue search instead of restarting
    if agent_loaded_successfully:
        logger.info("Resuming search")
        all_response, all_actions = agent.resume_predict(
            instruction,
            obs,
            search_metadata=search_metadata
        )
    else:
        logger.info("Starting fresh search")
        all_response, all_actions = agent.predict(
            instruction,
            obs,
            search_metadata=search_metadata
        )
    # done, run eval
    for response, actions in zip(all_response, all_actions):
        if len(actions) == 0:
            full_trajectory.append({"raw_action": response, "action": "None", "step_idx": step_idx})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})  # this will be same as last state
        
        ### this loop won't run anyway if no actions
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(
                action,
                env_args.sleep_after_execution
            )

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)

            # update trajectory
            full_trajectory.append({"raw_action": response, "action": action, "step_idx": step_idx})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})
            render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)
            
            # Save screenshot and trajectory information
            screenshot_fpath = os.path.join(result_dir, f"step_{step_idx + 1}_{action_timestamp}.png")
            with open(screenshot_fpath,"wb") as _fwrite:
                _fwrite.write(obs['screenshot'])
            with open(os.path.join(result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "raw_action": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    time_spent = time.time() - start_time
    logger.info(f"Time spent: {time_spent/60.0:.2f}min")

    ### first we save the trajectory
    traj_fpath = os.path.join(result_dir, "trajectory.pkl.xz")
    save_full_trajectory(full_trajectory, traj_fpath)

    ### compute score and token usage
    result = env.evaluate()
    result = float(result)
    logger.info(f"Result: {result:.2f}")

    # replaying actions again, as sometimes environment have randomness
    if result != 1.0:
        for _ in range(2):
            logger.info(f"Replaying found actions again...")
            result = _eval_again(
                env,
                env_args,
                task_config,
                result_dir,
                agent,
                all_response,
                all_actions
            )
            if result == 1.0:
                break
        logger.info(f"Replaying finished. Result: {result:.2f}")

    all_token_usages_so_far = get_all_token_usage()
    logger.info(f"Token consumption: {all_token_usages_so_far})")

    with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as fwrite:
        fwrite.write(f"{result}\n")
    with open(os.path.join(result_dir, "performance.json"), "w") as fwrite:
        perf = {
            "score": result,
            "llm_token": all_token_usages_so_far,
            "time_spent": time_spent/60.0,
        }
        json.dump(perf, indent=4, sort_keys=True, fp=fwrite)
    
    ### end recording
    if env_args.save_recording:
        env.controller.end_recording(os.path.join(result_dir, "recording.mp4"))
    ### all done. maybe save agent state anyway
    if common_args.save_agent_state:
        logger.info("Run finished normally. Saving the agent state")
        agent.save_state(agent_fpath)
    elif os.path.exists(agent_fpath):
        logger.info("Run finished normally. Removing the agent state")
        os.remove(agent_fpath)
    
    return {
        'score': result,
        'llm_token': all_token_usages_so_far,
        'time_spent': time_spent/60.0,
    }


def run_single_resumable_reinforced_search(
    agent: BaseAgent | RAgentMixin | ResumableAgentMixin,
    env: PooledDesktopEnv,
    task_config: dict,
    common_args: CommonArgs,
    env_args: EnvArgs,
    agent_args: AgentArgs,
    result_dir: str
):
    instruction = task_config["instruction"]
    ##### init env, agent, and etc.
    reset_token_usage()
    agent.reset()

    # ragent hook
    agent.on_task_start(task_config)

    logger.info(f"Running experiment with {env.cache_dir_root=}")
    obs = env.reset(task_config=task_config)
    done = False
    info = None
    reward = 0.0
    
    if env_args.save_recording:
        env.controller.start_recording()
    
    full_trajectory = [
        {"obs": obs, "info": info, "reward": reward, "done": done},
    ]
    html_fpath = os.path.join(result_dir, "trajectory.html")
    render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)

    ##### action loop
    start_time = time.time()
    search_metadata = construct_search_metadata(
        agent_name=agent.name,
        result_dir=result_dir,
        env=env,
        env_args=env_args,
        task_config=task_config,
    )
    # unlike normal agent, search agent directly output the final TRAJECTORY of actions
    step_idx = 0
    agent_fpath = os.path.join(result_dir, "agent.state")
    agent_loaded_successfully = _try_loading_agent_state_if_exists(agent, agent_fpath)
    
    # load agent state and continue search instead of restarting
    if agent_loaded_successfully:
        logger.info("Resuming search")
        all_response, all_actions = agent.resume_predict(
            instruction,
            obs,
            search_metadata=search_metadata
        )
    else:
        logger.info("Starting fresh search")
        all_response, all_actions = agent.predict(
            instruction,
            obs,
            search_metadata=search_metadata
        )
    # done, run eval
    for response, actions in zip(all_response, all_actions):
        if len(actions) == 0:
            full_trajectory.append({"raw_action": response, "action": "None", "step_idx": step_idx})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})  # this will be same as last state
        
        ### this loop won't run anyway if no actions
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(
                action,
                env_args.sleep_after_execution
            )

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)

            # update trajectory
            full_trajectory.append({"raw_action": response, "action": action, "step_idx": step_idx})
            full_trajectory.append({"obs": obs, "info": info, "reward": reward, "done": done})
            render_trajectory_to_html(task_config, full_trajectory, agent.obs_processor, html_fpath)
            
            # Save screenshot and trajectory information
            screenshot_fpath = os.path.join(result_dir, f"step_{step_idx + 1}_{action_timestamp}.png")
            with open(screenshot_fpath,"wb") as _fwrite:
                _fwrite.write(obs['screenshot'])
            with open(os.path.join(result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "raw_action": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    time_spent = time.time() - start_time
    logger.info(f"Time spent: {time_spent/60.0:.2f}min")

    ### first we save the trajectory
    traj_fpath = os.path.join(result_dir, "trajectory.pkl.xz")
    save_full_trajectory(full_trajectory, traj_fpath)

    ### compute score and token usage
    result = env.evaluate()
    result = float(result)
    logger.info(f"Result: {result:.2f}")

    # replaying actions again, as sometimes environment have randomness
    if result != 1.0:
        for _ in range(2):
            logger.info(f"Replaying found actions again...")
            result = _eval_again(
                env,
                env_args,
                task_config,
                result_dir,
                agent,
                all_response,
                all_actions
            )
            if result == 1.0:
                break
        logger.info(f"Replaying finished. Result: {result:.2f}")

    ### agent on task end
    agent.on_task_end(full_trajectory, task_config, {'success': 1.0 if result == 1.0 else 0.0})

    all_token_usages_so_far = get_all_token_usage()
    logger.info(f"Token consumption: {all_token_usages_so_far})")

    with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as fwrite:
        fwrite.write(f"{result}\n")
    with open(os.path.join(result_dir, "performance.json"), "w") as fwrite:
        perf = {
            "score": result,
            "llm_token": all_token_usages_so_far,
            "time_spent": time_spent/60.0,
        }
        json.dump(perf, indent=4, sort_keys=True, fp=fwrite)
    
    ### end recording
    if env_args.save_recording:
        env.controller.end_recording(os.path.join(result_dir, "recording.mp4"))
    ### all done. maybe save agent state anyway
    if common_args.save_agent_state:
        logger.info("Run finished normally. Saving the agent state")
        agent.save_state(agent_fpath)
    elif os.path.exists(agent_fpath):
        logger.info("Run finished normally. Removing the agent state")
        os.remove(agent_fpath)
    
    return {
        'score': result,
        'llm_token': all_token_usages_so_far,
        'time_spent': time_spent/60.0,
    }
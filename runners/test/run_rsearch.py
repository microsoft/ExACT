import datetime
import json
import logging
import os
import shutil
import traceback
from transformers import HfArgumentParser
from tqdm import tqdm
from exact.env.desktop_env_dev import DynamicPooledDesktopEnv
from exact.agent.agent_factory import get_agent_arg_cls, get_value_arg_cls, construct_agent, construct_value_function
from exact.logging import setup_logger
from exact.run_utils import run_single_reinforced_search
from exact.llms.utils import is_vlm
from exact.args import CommonArgs, EnvArgs, AgentArgs, ValueArgs
from dataclasses import asdict


logger = logging.getLogger("src.experiment")
pure_text_settings = ['a11y_tree']


def config() -> tuple[CommonArgs, EnvArgs, AgentArgs, ValueArgs]:
    # parse once to get the agent type
    parser = HfArgumentParser((CommonArgs, EnvArgs, AgentArgs, ValueArgs))
    _, _, agent_args, value_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    print(f'Running agent: {agent_args.agent}')

    # real parse
    agent_args_cls = get_agent_arg_cls(agent_args.agent)
    value_func_cls = get_value_arg_cls(value_args.value_func)
    parser = HfArgumentParser((CommonArgs, EnvArgs, agent_args_cls, value_func_cls))
    common_args, env_args, agent_args, vfunc_args = parser.parse_args_into_dataclasses()

    if not is_vlm(agent_args.model):
        assert env_args.observation_type in pure_text_settings, f"The model {agent_args.model} can only support text-based input, please consider change based model or settings"
    return common_args, env_args, agent_args, vfunc_args


def test(
    common_args: CommonArgs,
    env_args: EnvArgs,
    agent_args: AgentArgs,
    vfunc_args: ValueArgs,
    main_results_dir: str,
    test_all_meta: dict
) -> None:
    scores = []
    llm_tokens = []
    time_spent = []

    # log args
    logger.info(f"[Common Args]: {common_args}")
    logger.info(f"[Env Args]: {env_args}")
    logger.info(f"[Agent Args]: {agent_args}")
    # set wandb project
    cfg_args = {
        "common_args": asdict(common_args),
        "env_args": asdict(env_args),
        "agent_args": asdict(agent_args)
    }

    value_func = construct_value_function(
        vfunc_args.value_func,
        vfunc_args,
        observation_type=env_args.observation_type,
        action_space=env_args.action_space
    )
    agent = construct_agent(
        agent_args.agent,
        agent_args,
        env_args.action_space,
        env_args.observation_type,
        value_func=value_func
    )

    n_instances_per_task = env_args.n_sim_instances
    assert n_instances_per_task > 0, "n_sim_instances must be greater than 0 UNLESS you are doing one-shot ReACT"
    logger.info(f"setting env n_instances_per_task to {n_instances_per_task}")
    env = DynamicPooledDesktopEnv(
        # path_to_vm=args.path_to_vm,
        # action_space=agent.action_space,
        provider_name="docker",
        action_space=env_args.action_space,
        cache_dir=env_args.cache_dir,
        screen_size=(env_args.screen_width, env_args.screen_height),
        headless=env_args.headless,
        require_a11y_tree=env_args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
        n_instances_per_task=n_instances_per_task,
    )

    for domain in tqdm(test_all_meta, desc="Domain"):
        for example_id in tqdm(test_all_meta[domain], desc="Example", leave=False):
            config_file = os.path.join(
                common_args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
            )
            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)

            logger.info(f"[Domain]: {domain}")
            logger.info(f"[Example ID]: {example_id}")

            instruction = example["instruction"]

            logger.info(f"[Instruction]: {instruction}")
            # wandb each example config settings
            cfg_args["instruction"] = instruction
            cfg_args["start_time"] = datetime.datetime.now().strftime(
                "%Y:%m:%d-%H:%M:%S"
            )

            
            example_result_dir = os.path.join(
                main_results_dir,
                domain,
                example_id,
            )
            example_log_dir = os.path.join(example_result_dir, "logs")
            os.makedirs(example_result_dir, exist_ok=True)
            os.makedirs(example_log_dir, exist_ok=True)

            setup_logger(example_log_dir)
            # example start running
            logger.info(f"Evaluating {domain}/{example_id}")
            try:
                perf_dict = run_single_reinforced_search(
                    agent,
                    env,
                    example,
                    common_args,
                    env_args,
                    agent_args,
                    example_result_dir
                )
                scores.append(perf_dict["score"])
                llm_tokens.append(perf_dict["llm_token"])
                time_spent.append(perf_dict["time_spent"])
            except Exception as e:
                logger.error(f"Exception in {domain}/{example_id}: {e}")
                if env_args.save_recording:
                    env.controller.end_recording(
                        os.path.join(example_result_dir, "recording.mp4")
                    )
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(
                        json.dumps(
                            {"Error": f"Time limit exceeded in {domain}/{example_id}"}
                        )
                    )
                    f.write("\n")
                # save the exception
                error_text = traceback.format_exc()
                with open(os.path.join(example_result_dir, "error.txt"), "w") as f:
                    f.write(error_text)

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")
    logger.info(f"Average time spent: {sum(time_spent) / len(time_spent)}")
    logger.info(f"Results are now available under {main_results_dir}")
    return


def get_unfinished(
    target_dir, total_file_json
):
    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        if 'logs' in domain or domain.endswith(".json"):
            continue
        
        if domain not in total_file_json:
            ## do not touch stuff that we are not going to run
            continue

        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                if example_id not in total_file_json[domain]:
                    continue

                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        # empty all files under example_id
                        logger.info(f"Did not find performance under: {domain}/{example_id}. Removing")
                        shutil.rmtree(example_path)
                    else:
                        finished[domain].append(example_id)

    for domain, finished_examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in finished_examples
            ]
    return total_file_json


if __name__ == "__main__":
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    common_args, env_args, agent_args, vfunc_args = config()

    main_results_dir = os.path.join(
        common_args.result_dir,
        agent_args.agent,
        f"{common_args.exp_name}__{agent_args.model_id}__{env_args.action_space}__{env_args.observation_type}",
    )
    os.makedirs(main_results_dir, exist_ok=True)
    if agent_args.db_path is None:
        agent_args.db_path = os.path.join(main_results_dir, "db")
    if vfunc_args.vf_db_path is None:
        vfunc_args.vf_db_path = os.path.join(main_results_dir, "db")
    with open(os.path.join(main_results_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump({
            "common_args": asdict(common_args),
            "env_args": asdict(env_args),
            "agent_args": asdict(agent_args),
            "vfunc_args": asdict(vfunc_args)
        }, f, indent=4, sort_keys=True)

    #### setup logger
    log_dir = os.path.join(main_results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    setup_logger(log_dir)

    #### check task to run
    with open(common_args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)
    if common_args.domain != "all":
        test_all_meta = {common_args.domain: test_all_meta[common_args.domain]}

    test_file_list = get_unfinished(main_results_dir, test_all_meta)
    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info(f"Left tasks:\n{left_info}")


    #### run the test
    logger.info(f"Storing results under {main_results_dir=}")
    test(common_args, env_args, agent_args, vfunc_args, main_results_dir, test_file_list)

import logging
import subprocess
import argparse
import os
import time
import sys
import torch
import openai
import lzma
import pickle
import shutil
from runners.utils.prepare_vwa import EnvNames, EnvStatus, ENV_RESET_TIMEOUT, get_env_status
from runners.utils.prepare_vwa import main as prepare_vwa_main
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from transformers import HfArgumentParser
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
# environment variables
from configs.llms.providers import AVAILABLE_API_PROVIDERS, _API_PROVIDERS
from browser_env.env_config import (
    CLASSIFIEDS,
    CLASSIFIEDS_RESET_TOKEN,
    GITLAB,
    HOMEPAGE,
    MAP,
    REDDIT,
    SHOPPING,
    SHOPPING_ADMIN,
    WIKIPEDIA,
)


ENV_URLS_SH = f"""
export CLASSIFIEDS="{CLASSIFIEDS}"
export CLASSIFIEDS_RESET_TOKEN="{CLASSIFIEDS_RESET_TOKEN}"
export SHOPPING="{SHOPPING}"
export REDDIT="{REDDIT}"
export WIKIPEDIA="{WIKIPEDIA}"
export SHOPPING_ADMIN="{SHOPPING_ADMIN}"
export GITLAB="{GITLAB}"
export MAP="{MAP}"
export HOMEPAGE="{HOMEPAGE}"
""".strip()


assert os.environ.get("OPENAI_API_KEY") is not None, "Please set OPENAI_API_KEY"
assert os.environ.get("AZURE_TOKEN_PROVIDER_API_BASE") is not None, "Please set AZURE_TOKEN_PROVIDER_API_BASE"


logger = logging.getLogger(__name__)


@dataclass
class CommonArgs:
    # other args should be configured by the base run shell script
    env_name: str = field(
        metadata={
            "help": "The environment names to evaluate.",
            "choices": [
                EnvNames.classifields,
                EnvNames.cms,
                EnvNames.gitlab,
                EnvNames.reddit,
                EnvNames.shopping,
                EnvNames.wikipedia,
            ],
        }
    )
    save_dir: str = field(
        metadata={
            "help": "The directory to save the output files."
        }
    )
    eval_script: str = field(
        metadata={
            "help": "The path to the base evaluation shell script."
        }
    )
    # behavior of this python script
    run_mode: str = field(
        metadata={
            "help": "The running mode of the evaluation script.",
            "choices": ["polite", "greedy"],
        }
    )
    num_parallel: int = field(
        metadata={
            "help": "The number of parallel evaluations."
        }
    )
    main_api_providers: str = field(
        metadata={
            "help": "The main API providers to use. Needs to be the same length (separated by ,) as the number of parallel evaluations."
        }
    )
    num_task_per_script: int = field(
        metadata={
            "help": "The number of tasks per script run."
        }
    )
    num_task_per_reset: int = field(
        metadata={
            "help": "The number of tasks per environment reset. This is relevant if you use the GREEDY run mode."
        }
    )

    ## optionals
    start_idx: int = field(
        default=0,
        metadata={
            "help": "The start index of the evaluation."
        }
    )
    end_idx: int = field(
        default=-1,
        metadata={
            "help": "The end index of the evaluation."
        }
    )
    test_indices: str = field(
        default="",
        metadata={
            "help": "The test indices to evaluate. Use comma to separate multiple indices."
        }
    )

    def _check_base_script(self):
        base_script_file = self.eval_script
        assert os.path.exists(base_script_file), f"Cannot find the base script file: {base_script_file}"

        with open(base_script_file, 'r', encoding='utf-8') as fread:
            base_script_content = fread.read()

        assert "[[[API_PROVIDER_ENV_VARS]]]" in base_script_content
        assert "[[[test_idx]]]" in base_script_content
        assert "[[[SAVE_ROOT_DIR]]]" in base_script_content
        if self.env_name in ["gitlab", "cms"]:
            # webarena
            if self.env_name == "cms":
                assert f"configs/webarena/test_shopping_admin" in base_script_content, "Is your test config correctly setup?"
            else:
                assert f"configs/webarena/test_{self.env_name}" in base_script_content, "Is your test config correctly setup?"
        else:
            # visualwebarena
            if self.env_name == "classifields":
                assert f"configs/visualwebarena/test_classifieds" in base_script_content, "Is your test config correctly setup?"
            else:
                assert f"configs/visualwebarena/test_{self.env_name}" in base_script_content, "Is your test config correctly setup?"
        return

    def __post_init__(self):
        # e.g., with 3 paralle, you can do ["azure", "sglang", "openai"]
        if isinstance(self.main_api_providers, str):
            self.main_api_providers = self.main_api_providers.split(",")
        assert len(self.main_api_providers) == self.num_parallel, f"Received {self.main_api_providers=} but has {self.num_parallel=}"

        assert self.num_task_per_script <= 3, "Please set num_task_per_script <= 3 for reset to work nicely"
        assert self.num_task_per_reset > self.num_task_per_script, f"Received {self.num_task_per_reset=} but has {self.num_task_per_script=}"

        # check test_indices or start_idx, end_idx
        if self.end_idx == -1 and self.test_indices == "":
            raise ValueError("Please set either end_idx or test_indices.")
        if self.test_indices != "" and self.end_idx != -1:
            raise ValueError("Please set either end_idx or test_indices, not both.")
        if self.test_indices != "":
            self.test_indices = self.test_indices.split(",")
            self.test_indices = [int(idx) for idx in self.test_indices]

        # base script needs have the following being replacable
        self._check_base_script()
        return


def setup_logger(log_folder: str):    
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(
        log_folder,
        f"vwa_parallel_runner.log.txt"
    )

    formatting = "[%(asctime)s] %(levelname)s@%(name)s [%(pathname)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format=formatting,
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    return


def is_cuda_available():
    return torch.cuda.is_available()


def is_provider_alive(provider: str):
    if "azure" in provider:
        endpoint = _API_PROVIDERS['azure']['llm_api_base']
        azure_credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            azure_credential,
            os.environ.get("AZURE_TOKEN_PROVIDER_API_BASE", default="")
        )
        client = AzureOpenAI(
            api_version=_API_PROVIDERS['azure']['llm_api_version'],
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider
        )
        try:
            _ = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": "what are the last 4 digits of the sum of the first 70 prime numbers?",
                    },
                ],
                max_tokens=128,
                timeout=60
            )
        except Exception as e:
            logger.error(f"Provider {provider} is not accessible. Error: {e}")
            return False
    elif "sglang" in provider:
        # sglang is always alive
        endpoint = _API_PROVIDERS['sglang']['llm_api_base']
        client = openai.Client(api_key="sk-123456", base_url=endpoint)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/test/lang/example_image.png"
                        },
                        "modalities": "multi-images",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png"
                        },
                        "modalities": "multi-images",
                    },
                    {
                        "type": "text",
                        "text": "I have two very different images. They are not related at all. "
                        "Please describe the first image in one sentence, and then describe the second image in another sentence.",
                    },
                ],
            }
        ]
        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=1.0,
            top_p=0.7,
            n=2
        )

        assert response.choices[0].message.role == "assistant"
        assert len(response.choices) == 2
        return True
    else:
        # openai is always alive
        return True
    return True


def _wait_for_env_reset(env_name: str):
    # while loop to wait running new scipts when the env is resetting
    max_time_to_wait = ENV_RESET_TIMEOUT
    time_elapsed = 0
    while time_elapsed < max_time_to_wait:
        time.sleep(10)
        time_elapsed += 10
        status = get_env_status(env_name)
        if status == EnvStatus.resetting:
            logger.info(f"Environment {env_name} is still resetting. Continue waiting")
        else:
            break

    # final check
    status = get_env_status(env_name)
    if status == EnvStatus.resetting:
        logger.warning(f"Environment {env_name} is STILL resetting after {time_elapsed/60} min.")
    logger.info(f"Environment {env_name} is {status} now. Running evaluation scripts.")
    return


def _get_existing_tids(base_folder_path: str):
    found_tids = []
    for file in os.listdir(base_folder_path):
        if file.startswith("render_") and file.endswith(".html"):
            task_id = file.split(".")[0].split("_")[1]
            found_tids.append(int(task_id))
    logger.info(f"Found {len(found_tids)} existing tasks. Skipping {found_tids=}.")
    return set(found_tids)


def remove_existing_tids(indicies_to_run: list[int], base_folder_path: str):
    existing_tids = _get_existing_tids(base_folder_path)
    new_tids = set(indicies_to_run) - existing_tids
    return list(new_tids)


def check_if_run_failed(task_id: int, base_folder_path: str):
    error_rerun_dir = os.path.join(base_folder_path, "error_tids")
    if not os.path.exists(error_rerun_dir):
        logger.info(f"Error rerun directory {error_rerun_dir} does not exist. Assuming no failed tasks.")
        return False
    failed_file = os.path.join(error_rerun_dir, f"{task_id}.txt")
    if os.path.exists(failed_file):
        logger.info(f"Task {task_id} has failed with {failed_file}.")
        return True
    return False


def _load_all_task_records(base_folder_path: str):
    all_task_records = {}
    task_record_path = os.path.join(base_folder_path, "db", "task_records")
    logger.info(f"Loading all task records from {task_record_path}")
    if not os.path.exists(task_record_path):
        return {}
    
    for file in os.listdir(task_record_path):
        with lzma.open(os.path.join(task_record_path, file), "rb") as fread:
            task_record = pickle.load(fread)
        task_record_hash = hash(task_record)
        all_task_records[task_record_hash] = task_record
    return all_task_records


def remove_all_data_from_task(task_ids: list[int], base_folder_path: str, all_task_records: dict):
    # used when a task went into error, and we want to rerun it
    ### remove stuff from db
    # first retrieve task record so we know which reflections to remove
    if len(all_task_records) != 0:
        # this only exists for RMCTS
        logger.info(f"Using provided all_task_records with {len(all_task_records)=}.")

        found_task_hashes = []
        policy_reflections_path = os.path.join(base_folder_path, "db", "policy_reflections")
        for file in os.listdir(policy_reflections_path):
            policy_refl_file = os.path.join(policy_reflections_path, file)
            with lzma.open(policy_refl_file, "rb") as fread:
                reflections = pickle.load(fread)
            task_hash = reflections._from_task_hash
            corresponding_task = all_task_records.get(task_hash, None)
            if corresponding_task is None:
                logger.error(f"Could not find task record {task_hash} for policy reflection {policy_refl_file}.")
                continue

            refl_tid = corresponding_task.task_info['task_id']
            if int(refl_tid) in task_ids:
                found_task_hashes.append(task_hash)
                logger.debug(f"Removing policy reflection file {policy_refl_file}")
                if os.path.exists(policy_refl_file):
                    os.remove(policy_refl_file)
                break

        value_reflections_path = os.path.join(base_folder_path, "db", "value_reflections")
        for file in os.listdir(value_reflections_path):
            value_refl_file = os.path.join(value_reflections_path, file)
            with lzma.open(value_refl_file, "rb") as fread:
                reflections = pickle.load(fread)
            task_hash = reflections._from_task_hash
            corresponding_task = all_task_records.get(task_hash, None)
            if corresponding_task is None:
                logger.error(f"Could not find task record {task_hash} for value reflection file {value_refl_file}")
                continue

            refl_tid = corresponding_task.task_info['task_id']
            if int(refl_tid) in task_ids:
                found_task_hashes.append(task_hash)
                logger.debug(f"Removing value reflection file {value_refl_file}")
                if os.path.exists(value_refl_file):
                    os.remove(value_refl_file)
                break
        
        # finally, remove the task record itself
        for found_task_hash in found_task_hashes:
            task_record_file = os.path.join(base_folder_path, "db", "task_records", f"{found_task_hash}.pkl.xz")
            logger.debug(f"Removing task record file {task_record_file}")
            if os.path.exists(task_record_file):
                os.remove(task_record_file)
    

    search_tree_folder = os.path.join(base_folder_path, "search_trees")
    error_rerun_dir = os.path.join(base_folder_path, "error_tids")
    for task_id in task_ids:
        ### remove render
        render_file_path = os.path.join(base_folder_path, f"render_{task_id}.html")
        logger.info(f"Removing render file {render_file_path}")
        if os.path.exists(render_file_path):
            os.remove(render_file_path)
        #### remove stuff from search trees
        search_tree_task_folder = os.path.join(search_tree_folder, f"task_{task_id}")
        logger.info(f"Removing search tree folder {search_tree_task_folder}")
        if os.path.exists(search_tree_task_folder):
            shutil.rmtree(search_tree_task_folder)
        #### remove error file
        error_file = os.path.join(error_rerun_dir, f"{task_id}.txt")
        logger.info(f"Removing error file {error_file}")
        if os.path.exists(error_file):
            os.remove(error_file)
    
    #### other stuff such as log and performance will be overwritten once the task is rerun
    return


def run_single_script(env_name: str, test_indices: list[int], provider: str, save_dir: str, eval_script_path: str):
    if not is_cuda_available():
        logger.error(f"Torch is no longer available. Skipping this run.")
        return {
            'process.returncode': -100,
            'log_file': None,
            'test_indices': test_indices,
            'provider': provider
        }
    if not is_provider_alive(provider):
        logger.error(f"Provider {provider} is not accessible. Skipping this run.")
        return {
            'process.returncode': -100,
            'log_file': None,
            'test_indices': test_indices,
            'provider': provider
        }
    _wait_for_env_reset(env_name)
    
    # inject provider into script for better reproducibility
    eval_shell_command = ""
    with open(eval_script_path, 'r', encoding='utf-8') as fread:
        eval_shell_command = fread.read()

    # replace the necessary keys
    eval_shell_command = eval_shell_command.replace("[[[API_PROVIDER_ENV_VARS]]]", AVAILABLE_API_PROVIDERS[provider])
    eval_shell_command = eval_shell_command.replace("[[[test_idx]]]", ",".join([str(idx) for idx in test_indices]))
    eval_shell_command = eval_shell_command.replace("[[[SAVE_ROOT_DIR]]]", save_dir)

    # save this script
    ran_indices_str = "-".join([str(idx) for idx in test_indices])
    script_save_path = os.path.join(save_dir, "parallel_runner", f"run_{ran_indices_str}.sh")

    with open(script_save_path, 'w', encoding='utf-8') as fwrite:
        fwrite.write(eval_shell_command)
    # change the permission
    os.chmod(script_save_path, 0o755)

    script_output_path = os.path.join(save_dir, "parallel_runner", f"run_{ran_indices_str}.output.txt")
    with open(script_output_path, 'w', encoding='utf-8') as log_file:
        pass
    log_file = open(script_output_path, 'a', encoding='utf-8')

    logger.info(f"Running shell script created at: {script_save_path}")

    process = subprocess.Popen(
        f"sh {script_save_path}",
        shell=True,
        start_new_session=True,
        stdin=log_file,
        stdout=log_file,
        stderr=log_file,
        text=True
    )
    logger.info(f"{script_save_path} is running with pid: {process.pid}")

    # block until the script finishes
    process.wait()
    log_file.close()
    return {
        'process.returncode': process.returncode,
        'log_file': log_file,
        'test_indices': test_indices,
        'provider': provider
    }


def get_task_indices_to_run(remaining_indices: list[int], num_parallel: int, num_task_per_script: int):
    task_indices_to_run = []
    for i in range(num_parallel):
        task_indices = remaining_indices[i * num_task_per_script: (i + 1) * num_task_per_script]
        task_indices_to_run.append(task_indices)
    return task_indices_to_run


def get_providers_to_run(all_api_providers: list[str], num_parallel: int):
    provider_to_run = []
    for i in range(num_parallel):
        provider = all_api_providers[i]
        provider_to_run.append(provider)
    return provider_to_run


def reset_env(env_name: str):
    # reset the environment
    logger.info(f"Resetting the environment {env_name}.")
    arg = argparse.Namespace(
        mode="reset",
        env=env_name,
        force=True
    )
    prepare_vwa_main(arg)
    logger.info(f"Done resetting the environment {env_name}.")
    return


def refresh_env_login():
    logger.info("Refreshing login tokens.")
    os.makedirs("./.auth", exist_ok=True)

    dataset = os.environ.get("DATASET")
    assert dataset in ["webarena", "visualwebarena"], f"Unknown dataset {dataset=}"
    if dataset == "visualwebarena":
        urls = f"""
        export DATASET=visualwebarena
        {ENV_URLS_SH}
        """.replace(" "*4, "").strip()
    else:
        urls = f"""
        export DATASET=webarena
        {ENV_URLS_SH}
        """.replace(" "*4, "").strip()

    login_script_content = f"""
    {urls}

    python -m browser_env.auto_login
    """.replace(" "* 4, "").strip()

    login_script_path = "./.auth/refresh_login.sh"
    with open(login_script_path, 'w', encoding='utf-8') as fwrite:
        fwrite.write(login_script_content)

    process = subprocess.Popen(
        f"sh {login_script_path}",
        shell=True,
        start_new_session=True,
        text=True
    )
    process.wait()

    logger.info("Done refreshing login tokens.")
    return


def reserve_env(env_name: str):
    # reserve the environment
    logger.info(f"Reserving the environment {env_name}.")
    arg = argparse.Namespace(
        mode="reserve",
        env=env_name
    )
    prepare_vwa_main(arg)
    logger.info(f"Done reserving the environment {env_name}.")
    return


def free_env(env_name: str):
    # free the environment
    logger.info(f"Freeing the environment {env_name}.")
    arg = argparse.Namespace(
        mode="free",
        env=env_name
    )
    prepare_vwa_main(arg)
    logger.info(f"Done freeing the environment {env_name}.")
    return


def polite_parallel_run(args: CommonArgs):
    # let each script run k=3 tasks
    # when all of the N script finished, it should finish kN tasks
    # do:
    # 1. reset environment
    # 2. reserve eval
    # 3. do login
    # 4. repeat N scripts eval
    # 5. free
    if args.end_idx == -1:
        all_task_indices = args.test_indices
    else:
        all_task_indices = list(range(args.start_idx, args.end_idx))
    all_task_indices = remove_existing_tids(all_task_indices, args.save_dir)
    
    finished_indices = set()
    prev_num_finished = 0
    refresh_env_login()
    reset_env(args.env_name)
    
    while len(finished_indices) < len(all_task_indices):
        remaining_indices = list(set(all_task_indices) - finished_indices)
        remaining_indices.sort()
        task_indices_to_run = get_task_indices_to_run(
            remaining_indices,
            num_parallel=args.num_parallel,
            num_task_per_script=args.num_task_per_script
        )
        provider_to_run = get_providers_to_run(
            args.main_api_providers,
            num_parallel=args.num_parallel
        )

        # env is reset by 1) reset, 2) refresh login tokens, 3) reserve eval
        if prev_num_finished >= args.num_task_per_reset:
            refresh_env_login()
            reset_env(args.env_name)
            prev_num_finished = 0
        reserve_env(args.env_name)
        # run eval
        with ThreadPoolExecutor(max_workers=args.num_parallel) as executor:
            futures = []
            for task_indices, api_provider in zip(task_indices_to_run, provider_to_run):
                if len(task_indices) == 0:
                    continue
                future = executor.submit(
                    run_single_script,
                    args.env_name,
                    task_indices,
                    api_provider,
                    args.save_dir,
                    args.eval_script
                )
                futures.append(future)

            for future in futures:
                result = future.result()
                finished_indices.update(result['test_indices'])
                if result['process.returncode'] != 0:
                    logger.error(f"Failed to run the script. Log file: {result['log_file']}")
                prev_num_finished += len(result['test_indices'])
        
        # done
        free_env(args.env_name)
        logger.info(f"Finished running {len(finished_indices)} out of {len(all_task_indices)} tasks.")
        logger.info(f"Remaining tasks: {set(all_task_indices) - finished_indices}")
    return


def polite_parallel_rerun_failed(args: CommonArgs):
    if args.end_idx == -1:
        all_task_indices = args.test_indices
    else:
        all_task_indices = list(range(args.start_idx, args.end_idx))
    
    failed_tids = []
    for tid in all_task_indices:
        if check_if_run_failed(tid, args.save_dir):
            failed_tids.append(tid)
    if len(failed_tids) == 0:
        logger.info("No failed tasks to rerun.")
        return
    
    all_task_records = _load_all_task_records(args.save_dir)
    remove_all_data_from_task(failed_tids, args.save_dir, all_task_records)

    # rerun these tasks again
    logger.info(f"Rerunning failed tasks: {failed_tids}")
    args.end_idx = -1
    args.test_indices = failed_tids
    polite_parallel_run(args)
    return


def get_all_run_task_indices(all_indices: list[int], num_parallel: int, num_task_per_script: int):
    finished_indices = set()
    all_scheduled_indices = []
    while len(finished_indices) < len(all_indices):
        remaining_indices = list(set(all_indices) - finished_indices)
        remaining_indices.sort()
        task_indices_to_run = get_task_indices_to_run(
            remaining_indices,
            num_parallel=num_parallel,
            num_task_per_script=num_task_per_script
        )
        for ran_indices in task_indices_to_run:
            if len(ran_indices) == 0:
                continue
            all_scheduled_indices.append(ran_indices)
            finished_indices.update(ran_indices)
    return all_scheduled_indices


def get_all_run_providers(all_api_providers: list[str], num_scheduled_runs: int):
    all_scheduled_providers = []
    for i in range(num_scheduled_runs):
        provider = all_api_providers[i % len(all_api_providers)]
        all_scheduled_providers.append(provider)
    return all_scheduled_providers


def greedy_parallel_run(args: CommonArgs):
    # similar to polite_parallel_run, start another script/resetting immediately after finishing one
    if args.end_idx == -1:
        all_task_indices = args.test_indices
    else:
        all_task_indices = list(range(args.start_idx, args.end_idx))
    all_task_indices = remove_existing_tids(all_task_indices, args.save_dir)
    
    prev_num_finished = 0
    finished_indices = set()
    refresh_env_login()
    reset_env(args.env_name)

    all_run_indices = get_all_run_task_indices(
        all_task_indices,
        num_parallel=args.num_parallel,
        num_task_per_script=args.num_task_per_script
    )
    all_run_providers = get_all_run_providers(
        args.main_api_providers,
        num_scheduled_runs=len(all_run_indices)
    )

    reserve_env(args.env_name)
    with ThreadPoolExecutor(max_workers=args.num_parallel) as executor:
        futures = []
        for task_indices, api_provider in zip(all_run_indices, all_run_providers):
            future = executor.submit(
                run_single_script,
                args.env_name,
                task_indices,
                api_provider,
                args.save_dir,
                args.eval_script
            )
            futures.append(future)

        for future in futures:
            result = future.result()
            finished_indices.update(result['test_indices'])
            if result['process.returncode'] != 0:
                logger.error(f"Failed to run the script. Log file: {result['log_file']}")

            logger.info(f"Finished running {len(finished_indices)} out of {len(all_task_indices)} tasks.")
            prev_num_finished += len(result['test_indices'])

            # run reset env
            # expect the next one is finishing soon
            check_failure = result['process.returncode'] == -100
            if prev_num_finished * 2 >= args.num_task_per_reset and not check_failure:
                refresh_env_login()
                reset_env(args.env_name)
                prev_num_finished = 0
    free_env(args.env_name)
    return


def greedy_parallel_rerun_failed(args: CommonArgs):
    if args.end_idx == -1:
        all_task_indices = args.test_indices
    else:
        all_task_indices = list(range(args.start_idx, args.end_idx))
    
    failed_tids = []
    for tid in all_task_indices:
        if check_if_run_failed(tid, args.save_dir):
            failed_tids.append(tid)
    if len(failed_tids) == 0:
        logger.info("No failed tasks to rerun.")
        return
    
    all_task_records = _load_all_task_records(args.save_dir)
    logger.info(f'removing {failed_tids}')
    remove_all_data_from_task(failed_tids, args.save_dir, all_task_records)

    # rerun these tasks again
    logger.info(f"Rerunning failed tasks: {failed_tids}")
    args.end_idx = -1
    args.test_indices = failed_tids
    greedy_parallel_run(args)
    return


if __name__ == "__main__":
    parser = HfArgumentParser((CommonArgs,))
    args, = parser.parse_args_into_dataclasses()

    print('Received args:', args)

    setup_logger(os.path.join(args.save_dir, "parallel_runner"))
    run_cmd = f"python {' '.join(sys.argv)}"
    logger.info(f'running [{run_cmd}]')

    start_time = time.time()
    if args.run_mode == "polite":
        polite_parallel_run(args)
        polite_parallel_rerun_failed(args)
    elif args.run_mode == "greedy":
        greedy_parallel_run(args)
        greedy_parallel_rerun_failed(args)
    else:
        print("Unknown run mode.")
    time_elapsed = time.time() - start_time
    logger.info(f"Command [{run_cmd}] finished running in {time_elapsed/3600} hours.")
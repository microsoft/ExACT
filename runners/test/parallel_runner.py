import logging
import subprocess
import os
import time
import sys
import openai
import yaml
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from transformers import HfArgumentParser
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


logger = logging.getLogger(__name__)


if os.environ.get("OPENAI_API_KEY", "") == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


@dataclass
class CommonArgs:
    # other args should be configured by the base run shell script
    eval_script: str = field(
        metadata={
            "help": "The path to the base evaluation shell script."
        }
    )
    test_name: str = field(
        metadata={
            "help": "The test name to use."
        }
    )
    # behavior of this python script
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
    run_tfiles: str = field(
        metadata={
            "help": "The test config files to run. Separated by ,."
        }
    )
    machine_name: str = field(
        metadata={
            "help": "Name of the machine running this script."
        }
    )
    save_dir: str = field(
        default=None,
        metadata={
            "help": "The result save directory. Will be inferred from the shell script."
        }
    )

    def _check_base_script(self):
        base_script_file = self.eval_script
        assert os.path.exists(base_script_file), f"Cannot find the base script file: {base_script_file}"

        with open(base_script_file, 'r', encoding='utf-8') as fread:
            base_script_content = fread.read()

        assert "[[[API_KEY_FPATH]]]" in base_script_content
        assert "[[[OSWORLD_DATA_DIR]]]" in base_script_content
        assert "[[[API_PROVIDER_ENV_VARS]]]" in base_script_content
        assert "[[[TEST_FPATH]]]" in base_script_content
        assert "[[[TEST_NAME]]]" in base_script_content
        return

    def _infer_save_dir(self):
        if self.save_dir is None:
            file_config_yaml = "runners/configs/files.yaml"
            with open(file_config_yaml, 'r', encoding='utf-8') as fread:
                file_configs = yaml.safe_load(fread)
            base_data_dir = file_configs[self.machine_name]['osworld_data_dir']
            result_base_dir = os.path.join(base_data_dir, "eval_results")
            
            with open(self.eval_script, 'r', encoding='utf-8') as fread:
                script_content = fread.read()
            ## parse the following enviornment variables
            for line in script_content.split("\n"):
                if "agent" in line and "$" not in line:
                    agent = line.split("=")[-1].strip()
                if "model_id" in line and "$" not in line:
                    model_id = line.split("=")[-1].strip()
                if "action_space" in line and "$" not in line:
                    action_space = line.split("=")[-1].strip()
                if "observation_type" in line and "$" not in line:
                    observation_type = line.split("=")[-1].strip()
            save_dir = os.path.join(
                result_base_dir,
                agent,
                f"{self.test_name}__{model_id}__{action_space}__{observation_type}",
            )
            os.makedirs(save_dir, exist_ok=True)
            self.save_dir = save_dir
            logger.info(f"Inferred save directory: {self.save_dir}")
        return

    def __post_init__(self):
        # e.g., with 3 paralle, you can do ["azure", "sglang", "openai"]
        if isinstance(self.main_api_providers, str):
            self.main_api_providers = self.main_api_providers.split(",")
        if isinstance(self.run_tfiles, str):
            self.run_tfiles = self.run_tfiles.split(",")
        assert len(self.main_api_providers) == self.num_parallel, f"Received {self.main_api_providers=} but has {self.num_parallel=}"
        assert len(self.run_tfiles) == self.num_parallel, f"Received {self.run_tfiles=} but has {self.num_parallel=}"

        # base script needs have the following being replacable
        self._check_base_script()
        self._infer_save_dir()
        return


def setup_logger(log_folder: str):
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(
        log_folder,
        f"parallel_runner.log.txt"
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


def is_provider_alive(provider: str):
    provider_config_yaml = "runners/configs/providers.yaml"
    with open(provider_config_yaml, 'r', encoding='utf-8') as fread:
        provider_configs = yaml.safe_load(fread)

    if "azure" in provider:
        endpoint = provider_configs[provider]['api_base']
        azure_credential = DefaultAzureCredential(
            exclude_managed_identity_credential=True,
        )
        token_provider = get_bearer_token_provider(
            azure_credential,
            provider_configs[provider]['token_provider_base']
        )
        client = AzureOpenAI(
            api_version=provider_configs[provider]['api_version'],
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
        endpoint = provider_configs[provider]['api_base']
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


def _to_bash_env_vars(env_vars: dict):
    provider = env_vars['provider']
    api_base = env_vars['api_base']
    api_key = env_vars['api_key']
    if api_key is None:
        api_key = os.environ['OPENAI_API_KEY']
    api_version = env_vars['api_version']
    token_provider_base = env_vars['token_provider_base']
    bash_export_str = f"""
    export POLICY_LLM_API_BASE="{api_base}"
    export POLICY_LLM_API_KEY="{api_key}"
    export POLICY_LLM_API_VERSION="{api_version}"
    export POLICY_LLM_TOKEN_PROVIDER_BASE="{token_provider_base}"

    export VALUE_LLM_API_BASE="{api_base}"
    export VALUE_LLM_API_KEY="{api_key}"
    export VALUE_LLM_API_VERSION="{api_version}"
    export VALUE_LLM_TOKEN_PROVIDER_BASE="{token_provider_base}"

    export REFLECTION_LLM_API_BASE="{api_base}"
    export REFLECTION_LLM_API_KEY="{api_key}"
    export REFLECTION_LLM_API_VERSION="{api_version}"
    export REFLECTION_LLM_TOKEN_PROVIDER_BASE="{token_provider_base}"

    model_api_provider={provider}
    vf_model_api_provider={provider}
    rlm_api_provider={provider}
    """.replace(" "*4, "").strip()
    return bash_export_str


def _fill_configs(test_name: str, test_fpath: str, provider: str, machine_name: str):
    file_config_yaml = "runners/configs/files.yaml"
    provider_config_yaml = "runners/configs/providers.yaml"
    with open(file_config_yaml, 'r', encoding='utf-8') as fread:
        file_configs = yaml.safe_load(fread)
    with open(provider_config_yaml, 'r', encoding='utf-8') as fread:
        provider_configs = yaml.safe_load(fread)
    
    filled_configs = {}
    filled_configs['api_key_fpath'] = file_configs[machine_name]['api_key_fpath']
    filled_configs['osworld_data_dir'] = file_configs[machine_name]['osworld_data_dir']
    filled_configs['api_provider_env_vars'] = _to_bash_env_vars(provider_configs[provider])
    filled_configs['test_fpath'] = test_fpath
    filled_configs['test_name'] = test_name
    return filled_configs


def run_single_script(test_fpath: str, provider: str, args: CommonArgs):
    # replace the necessary keys
    if not is_provider_alive(provider):
        logger.error(f"Provider {provider} is not accessible. Skipping this run.")
        return {
            'process.returncode': -100,
            'log_file': None,
            'test_fpath': test_fpath,
            'provider': provider
        }
    
    # inject provider into script
    machine_name = args.machine_name
    filled_configs = _fill_configs(args.test_name, test_fpath, provider, machine_name)

    eval_shell_command = ""
    with open(args.eval_script, 'r', encoding='utf-8') as fread:
        eval_shell_command = fread.read()
    eval_shell_command = eval_shell_command.replace("[[[API_KEY_FPATH]]]", filled_configs['api_key_fpath'])
    eval_shell_command = eval_shell_command.replace("[[[OSWORLD_DATA_DIR]]]", filled_configs['osworld_data_dir'])
    eval_shell_command = eval_shell_command.replace("[[[API_PROVIDER_ENV_VARS]]]", filled_configs['api_provider_env_vars'])
    eval_shell_command = eval_shell_command.replace("[[[TEST_FPATH]]]", filled_configs['test_fpath'])
    eval_shell_command = eval_shell_command.replace("[[[TEST_NAME]]]", filled_configs['test_name'])
    prun_save_dir = os.path.join(args.save_dir, "logs", "parallel_runner")

    # save this script
    ran_testfile_str = filled_configs['test_fpath'].split("/")[-1].replace(".json", "")
    script_save_path = os.path.join(prun_save_dir, f"run-{ran_testfile_str}.sh")

    with open(script_save_path, 'w', encoding='utf-8') as fwrite:
        fwrite.write(eval_shell_command)
    # change the permission
    os.chmod(script_save_path, 0o755)

    script_output_path = os.path.join(prun_save_dir, f"run-{ran_testfile_str}.output.txt")
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
        'test_fpath': test_fpath,
        'provider': provider
    }


def parallel_run(args: CommonArgs):
    all_run_tfiles = args.run_tfiles
    all_run_providers = args.main_api_providers
    with ThreadPoolExecutor(max_workers=args.num_parallel) as executor:
        futures = []
        for test_fpath, api_provider in zip(all_run_tfiles, all_run_providers):
            future = executor.submit(
                run_single_script,
                test_fpath,
                api_provider,
                args
            )
            futures.append(future)
            logger.info("Dispatched a new thread. Sleeping for 5 minutes before starting the next one.")
            time.sleep(5*60)  # when a program just starts, it can start a LOT of containers

        for f_idx, future in enumerate(futures):
            result = future.result()
            if result['process.returncode'] != 0:
                logger.error(f"Failed to run the script. Log file: {result['log_file']}")

            logger.info(f"Finished running {f_idx+1} out of {len(futures)} threads.")
    return


if __name__ == "__main__":
    parser = HfArgumentParser((CommonArgs,))
    args, = parser.parse_args_into_dataclasses()

    print('Received args:', args)

    prun_dir = os.path.join(args.save_dir, "logs", "parallel_runner")
    setup_logger(prun_dir)

    run_cmd = f"python {' '.join(sys.argv)}"
    logger.info(f'running [{run_cmd}]')

    start_time = time.time()
    
    parallel_run(args)
    parallel_run(args)   # run twice, as eval_script itself will only run failed ones

    time_elapsed = time.time() - start_time
    logger.info(f"Command [{run_cmd}] finished running in {time_elapsed/3600} hours.")
    logger.info(f"Result should be available at: {args.save_dir}")

from fastapi import FastAPI
from pydantic import BaseModel
import requests
import subprocess
import threading
import logging
import argparse
import uvicorn
import datetime
import sys
import os


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

file_handler = logging.FileHandler(f"distributed/logs/worker_{datetime_str}.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger = logging.getLogger(__name__)


# ensure shell scripts would work correctly
assert os.environ.get("OPENAI_API_KEY", None) is not None, "Please set OPENAI_API_KEY environment variable"


app = FastAPI()


class LocalRunConfig(BaseModel):
    base_shell_script: str
    test_name: str
    mconfig: dict


def run_python_cmd(cmd: str):
    try:
        subprocess.run(cmd, shell=True, check=True)
    except Exception as e:
        logger.error(f"Error running command: {cmd}")
        logger.error(e)
        return
    return


class Worker:
    def __init__(self, mname: str, api: str, manager_api: str):
        self.mname = mname
        self.api = api
        self.manager_api = manager_api

        self.register_worker()
        return

    def _compose_prunner_cmd(self, run_config: LocalRunConfig):
        base_shell_script = run_config.base_shell_script
        # create this shell script
        datetime_str: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shell_save_fpath = f"/tmp/osworld_prun/{self.mname}_{datetime_str}_run.sh"
        os.makedirs(os.path.dirname(shell_save_fpath), exist_ok=True)

        logger.info(f"Creating shell script at {self.mname} under:\n{base_shell_script}")
        with open(shell_save_fpath, "w") as fwrite:
            fwrite.write(base_shell_script)

        cmd = f"python runners/test/parallel_runner.py "
        cmd += f"--eval_script {shell_save_fpath} "
        cmd += f"--test_name {run_config.test_name} "
        cmd += f"--num_parallel {run_config.mconfig['num_parallel']} "
        cmd += f"--main_api_providers {','.join(run_config.mconfig['main_api_providers'])} "
        cmd += f"--run_tfiles {','.join(run_config.mconfig['run_tfiles'])} "
        cmd += f"--machine_name {self.mname} && rm {shell_save_fpath}"
        return cmd

    def run_parallel_runner(self, run_config: LocalRunConfig):
        logger.info(f"Running parallel runner with config: {run_config}")
        prunner_cmd = self._compose_prunner_cmd(run_config)
        logger.info(f"Received Command: {prunner_cmd}")
        # run this in a subprocess
        thread = threading.Thread(target=run_python_cmd, args=(prunner_cmd,))
        thread.start()

        thread_pid = thread.ident
        logger.info(f"Started thread with PID: {thread_pid}")
        return

    def register_worker(self):
        try:
            response = requests.post(
                f"http://{self.manager_api}/register_worker",
                params={
                    "mname": self.mname,
                    "api": self.api
                },
                timeout=20
            )
            resp = response.json()
            logger.info(f"Response: {resp}")
        except Exception as e:
            logger.error(f"Error registering worker: {e}")
            logger.info(f"If this machine can only receive, consider using distributed/dist_cli.py on manager machine.")
            return
        return


worker: Worker = None


@app.post("/run_parallel_runner")
def run_parallel_runner(run_config: LocalRunConfig):
    worker.run_parallel_runner(run_config)
    return {"status": "success"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mname", type=str, required=True)
    parser.add_argument("--addr_for_manager", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--manager_api", type=str, required=True)
    args = parser.parse_args()

    api = f"{args.addr_for_manager}:{args.port}"
    logger.info(f"Starting worker {args.mname} with API: {api}")

    worker = Worker(args.mname, api, args.manager_api)

    # start the FastAPI server
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=args.port,
        log_level="info"
    )
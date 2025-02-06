from fastapi import FastAPI
from pydantic import BaseModel
import requests
import yaml
import logging
import uvicorn
import datetime
import sys


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

file_handler = logging.FileHandler(f"distributed/logs/manager_{datetime_str}.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger = logging.getLogger(__name__)


app = FastAPI()


class RunConfig(BaseModel):
    base_shell_fpath: str
    test_name: str
    dist_config_path: str


class Manager:
    _mname_to_api = {}

    def __init__(self):
        pass

    def submit_job(self, mname: str, run_config: RunConfig, mconfig: dict):
        if mname not in self._mname_to_api:
            logger.error(f"Machine {mname} not found in the list of available machines")
            logger.info(f"Available machines: {self._mname_to_api.keys()}")
            return
        
        api = self._mname_to_api[mname]

        with open(run_config.base_shell_fpath, "r") as fread:
            base_shell_script = fread.read()
        
        # submit job
        logger.info(f"Submitting job to {mname} with config: {mconfig}")
        try:
            response = requests.post(
                f"http://{api}/run_parallel_runner",
                json={
                    "base_shell_script": base_shell_script,
                    "test_name": run_config.test_name,
                    "mconfig": mconfig
                },
                timeout=20
            )
            resp = response.json()
            logger.info(f"Response: {resp}")
        except Exception as e:
            logger.error(f"Error submitting job to {mname}: {e}")
            return
        
        return resp

    def register_worker(self, mname: str, api: str):
        self._mname_to_api[mname] = api
        logger.info(f"Current workers: {self._mname_to_api}")
        return


manager = Manager()


@app.post("/dist_run_parallel_runner")
async def dist_run_parallel_runner(
    run_config: RunConfig,
):
    with open(run_config.dist_config_path, "r") as f:
        per_machine_configs = yaml.safe_load(f)
    
    success = {}
    for mname, mconfig in per_machine_configs.items():
        print(f"Running {run_config.test_name} on {mname} with config: {mconfig}")
        resp = manager.submit_job(
            mname,
            run_config,
            mconfig
        )

        if resp is None:
            success[mname] = False
        else:
            success[mname] = resp
    return success


@app.post("/register_worker")
async def register_worker(
    mname: str,
    api: str
):
    manager.register_worker(mname, api)
    return {"message": f"Worker {mname} registered with api: {api}"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=12000,
        log_level="info"
    )
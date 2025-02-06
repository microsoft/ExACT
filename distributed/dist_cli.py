import requests
import argparse
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class DistRunArgs:
    manager_addr: str
    base_shell_fpath: str
    test_name: str
    dist_config_path: str

    mode: str = "dist_run_parallel_runner"

@dataclass
class RegisterWorkerArgs:
    mname: str
    addr_for_manager: str
    port: int
    manager_api: str

    mode: str = "register_worker"


def dist_run_parallel_runner(args: DistRunArgs):
    manager_addr = args.manager_addr
    resp = requests.post(
        f"http://{manager_addr}/dist_run_parallel_runner",
        json={
            "base_shell_fpath": args.base_shell_fpath,
            "test_name": args.test_name,
            "dist_config_path": args.dist_config_path
        },
        timeout=10
    )
    print(resp.json())
    return


def register_worker(args: RegisterWorkerArgs):
    resp = requests.post(
        f"http://{args.manager_api}/register_worker",
        params={
            "mname": args.mname,
            "api": f"{args.addr_for_manager}:{args.port}"
        },
        timeout=10
    )
    print(resp.json())
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="submit job")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dist_run_parallel_runner", "register_worker"],
        required=True
    )
    args, _ = parser.parse_known_args()
    if args.mode == "dist_run_parallel_runner":
        parser = HfArgumentParser(DistRunArgs)
        args,  = parser.parse_args_into_dataclasses()
        dist_run_parallel_runner(args)
    elif args.mode == "register_worker":
        parser = HfArgumentParser(RegisterWorkerArgs)
        args,  = parser.parse_args_into_dataclasses()
        register_worker(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

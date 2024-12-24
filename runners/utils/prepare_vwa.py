from gradio_client import Client
from src.envs.browser import FastCachedwActionMatchingBrowserEnv
import asyncio
import signal
import time
import argparse
import requests


client = Client("http://coffee.cs.columbia.edu:55405/")


class EnvNames:
    classifields = "classifields"
    cms = "cms"
    gitlab = "gitlab"
    reddit = "reddit"
    shopping = "shopping"
    wikipedia = "wikipedia"


class EnvStatus:
    resetting = "resetting"
    free = "free"
    evaluating = "evaluating"
    down = "down"


SPECIAL_CASES = {
    'classifields': {
        'CLASSIFIEDS': 'http://coffee.cs.columbia.edu:57981',
        'CLASSIFIEDS_RESET_TOKEN': '4b61655535e7ed388f0d40a93600254c'
    }
}


ENV_RESET_TIMEOUT = 10 * 60 # 10 minutes


def _get_env_status(status_str: str):
    if status_str.lower() == "resetting":
        return EnvStatus.resetting
    elif status_str.lower() == "free":
        return EnvStatus.free
    elif status_str.lower() == "evaluating":
        return EnvStatus.evaluating
    elif status_str.lower() == "down":
        return EnvStatus.down
    else:
        raise Exception(f"Unknown status {status_str}")


def get_env_status(env_name: str):
    result = client.predict(
        api_name="/refresh_log"
    )
    status_text = result[0]
    status_list_dict: list[dict] = result[1]

    print("Server log status")
    print("[START of server log]")
    print(status_text)
    print("[END of server log]")

    for status_dict in status_list_dict:
        if status_dict["token"].lower() == env_name.lower():
            return _get_env_status(status_dict["class_or_confidence"])
    raise Exception(f"Environment name {env_name} not found from tokens in {status_list_dict}.")
    return



EX_CONFIG_FILES = {
    EnvNames.classifields: "configs/visualwebarena/test_classifieds_intent_fixed/0.json",
    EnvNames.reddit: "configs/visualwebarena/test_reddit_intent_fixed/0.json",
    EnvNames.shopping: "configs/visualwebarena/test_shopping_intent_fixed/0.json",
}


def _handle_timeout(signum, frame):
    raise Exception("TimeoutError")


async def _is_env_running_fine(env_name: str):
    assert env_name in EX_CONFIG_FILES, f"Unknown env_name: {env_name} for checking asetup."
    config_file = EX_CONFIG_FILES[env_name]
    env = FastCachedwActionMatchingBrowserEnv(  # used specifically for search type algorithms
        headless=True,
        slow_mo=0,
        action_set_tag="som",
        observation_type="image_som",
        current_viewport_only=True,
        viewport_size={
            "width": 1280,
            "height": 2048,
        },
        save_trace_enabled=False,
        sleep_after_execution=2.5,
        # NOTE: captioning_fn here is used for LLM + captioning baselines.
        # This can be different from the captioning model used for evals.
        captioning_fn=None,
    )
    # do twice
    obs, info = await env.areset(options={"config_file": config_file})
    obs, info = await env.areset(options={"config_file": config_file})
    return


def is_env_running_fine(env_name: str):
    # check if setup can be returned in 60 seconds
    start_time = time.time()
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(60)
    try:
        asyncio.run(_is_env_running_fine(env_name))
    except Exception as e:
        raise e
    finally:
        signal.alarm(0)
    elased_time = time.time() - start_time
    print(f"Environment {env_name} reset call is running fine, returned in {elased_time:.2f}s.")
    return


def _fast_reset_classifields():
    # special case
    # first try a fast reset using curl
    # if it fails, then use the normal way
    try:
        request = requests.post(
            f"{SPECIAL_CASES['classifields']['CLASSIFIEDS']}/index.php?page=reset",
            data={"token": SPECIAL_CASES['classifields']['CLASSIFIEDS_RESET_TOKEN']},
            timeout=20.0  # do full reset if it takes too long to respond
        )
    except requests.exceptions.Timeout:
        return False
    
    if 'sqlError occurred' in request.text:
        return False

    # check is_env_running_fine
    try:
        _ = is_env_running_fine(EnvNames.classifields)
    except Exception as e:
        print(f"Testing {EnvNames.classifields} reset call failed. Try normal reset. Error: {e}")
        return False
    return True



def try_fast_resets(env_names: list[str]):
    succeeded_envs = []
    for env_name in env_names:
        if env_name == EnvNames.classifields:
            if _fast_reset_classifields():
                succeeded_envs.append(env_name)
                print(f"Fast reset for {env_name} succeeded.")
            else:
                print(f"Fast reset for {env_name} failed. Try normal reset.")
        else:
            print(f"Fast reset for {env_name} not implemented yet. Try normal reset.")

    for env in succeeded_envs:
        env_names.remove(env)
    return env_names


def free_env(env_name: str):
    result = client.predict(
        env_names=[env_name],
        api_name="/done_eval"
    )
    return result


def reserve_env(env_name: str):
    # while loop check if all envs are free
    max_time_to_wait = ENV_RESET_TIMEOUT
    time_elapsed = 0
    while time_elapsed < max_time_to_wait:
        time.sleep(10)
        time_elapsed += 10
        all_free = True
        
        status = get_env_status(env_name)
        if status == EnvStatus.down:
            raise Exception(f"Environment {env_name} is down. Cannot reserve it.")
        
        if status == EnvStatus.resetting:
            all_free = False
            print(f"Environment {env_name} is still resetting. Continue waiting")
            break
        if all_free:
            break

    if not all_free:
        status = get_env_status(env_name)
        raise Exception(f"Environment {env_name} is STILL {status} after {time_elapsed/60} min.")
    
    result = client.predict(
        env_names=[env_name],
        api_name="/reserve_eval"
    )
    return result


def _sync_reset(envs_to_reset: list[str]):
    # try fast resets
    original_envs_to_reset = envs_to_reset.copy()
    envs_to_reset = try_fast_resets(envs_to_reset)

    # reset all envs in parallel
    result = client.predict(
        env_to_reset=envs_to_reset,
        api_name="/reset_environment"
    )

    # while loop check if all envs are free
    max_time_to_wait = ENV_RESET_TIMEOUT
    time_elapsed = 0
    while time_elapsed < max_time_to_wait:
        time.sleep(10)
        time_elapsed += 10
        all_free = True
        for env in envs_to_reset:
            status = get_env_status(env)
            if status == EnvStatus.resetting:
                all_free = False
                print(f"Environment {env} is still resetting. Continue waiting")
                break
        if all_free:
            break

    # final check
    print("[Final check]")
    for env in original_envs_to_reset:
        status = get_env_status(env)
        if status == EnvStatus.resetting:
            raise Exception(f"Environment {env} is STILL resetting after {time_elapsed/60} min.")
        print(f"Environment {env} is {status} now.")
    return


def sync_reset(envs_to_reset: list[str], force: bool = False):
    print(f"Resetting environments {envs_to_reset} with {force=}")
    if force:
        _sync_reset(envs_to_reset)
        return
    
    # polite: check if each env is free before resetting
    for env in envs_to_reset:
        status = get_env_status(env)
        if status != EnvStatus.free:
            raise Exception(f"Environment {env} is not free, but {status}.")

    _sync_reset(envs_to_reset)
    return


def main(args: argparse.Namespace):
    if args.mode == "reset":
        status = get_env_status(args.env)
        print(f"Environment {args.env} is {status} now. Trying to {args.mode} it.")

        if status == EnvStatus.resetting:
            print(f"Environment {args.env} is already resetting. Skip.")
            return
        
        try:
            sync_reset([args.env], args.force)
        except Exception as e:
            print(f"Failed to reset {args.env}. Error: {e}")
            exit(1)
    elif args.mode == "free":
        status = get_env_status(args.env)
        print(f"Environment {args.env} is {status} now. Trying to {args.mode} it.")
        
        free_env(args.env)
    elif args.mode == "reserve":
        status = get_env_status(args.env)
        print(f"Environment {args.env} is {status} now. Trying to {args.mode} it.")
        
        reserve_env(args.env)
    else:
        print(f"Mode {args.mode} is not implemented yet.")
        exit(1)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare VWA environments")
    parser.add_argument(
        "mode",
        type=str,
        choices=["reserve", "free", "reset"],
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=[
            EnvNames.classifields,
            EnvNames.cms,
            EnvNames.gitlab,
            EnvNames.reddit,
            EnvNames.shopping,
            EnvNames.wikipedia,
        ],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force the operation regardless of the current status of the environment."
    )
    args = parser.parse_args()

    main(args)
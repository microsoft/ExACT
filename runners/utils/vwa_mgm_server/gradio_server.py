import argparse
import gradio as gr
import logging
import json
import subprocess
import requests
from typing import List
from dataclasses import dataclass, field, asdict
from src.logging import setup_logger
from browser_env.env_config import (
    CLASSIFIEDS,
    SHOPPING,
    REDDIT,
    WIKIPEDIA,
    CMS,
    GITLAB,
)


logger = logging.getLogger("vwa-server")
setup_logger(log_folder="logs")


headers = {"User-Agent": "VWA Management Client"}
args: argparse.Namespace


FREE = "free"
RESETING = "resetting"
EVALUATING = "evaluating"
DOWN = "down"
RESET_LOG_FILE = "logs/reset_log.log.txt"
RESET_BASH_FILES = {
    'classifields': 'shells/reset_classifieds.sh',
    'cms': 'shells/reset_cms.sh',
    'gitlab': 'shells/reset_gitlab.sh',
    'reddit': 'shells/reset_reddit.sh',
    'shopping': 'shells/reset_shopping.sh',
    'wikipedia': 'shells/reset_wikipedia.sh'
}
ENV_WEBSITE_URLS = {
    "classifields": CLASSIFIEDS,
    "shopping": SHOPPING,
    "reddit": REDDIT,
    "wikipedia": WIKIPEDIA,
    "cms": CMS,
    "gitlab": GITLAB,
}
SERVER_PORT = 55405


@dataclass
class ServerState:
    classifields_state: str = FREE
    cms_state: str = FREE
    gitlab_state: str = FREE
    reddit_state: str = FREE
    shopping_state: str = FREE
    wikipedia_state: str = FREE
    reset_info: dict = field(default_factory=dict)

    def get_status_text(self):
        return [
            ("Classifields", self.classifields_state),
            ("CMS", self.cms_state),
            ("GitLab", self.gitlab_state),
            ("Reddit", self.reddit_state),
            ("Shopping", self.shopping_state),
            ("Wikipedia", self.wikipedia_state),
        ]

    def update_env_state(self, envs: List[str], state: str):
        for env in envs:
            setattr(self, f"{env}_state", state)
        return

    def get_env_state(self, env: str):
        return getattr(self, f"{env}_state")

    def is_all_unavailable(self):
        is_all_unavailable = True
        for env in ServerState.get_env_names():
            if self.get_env_state(env) == FREE:
                is_all_unavailable = False
                break
        return is_all_unavailable

    @staticmethod
    def get_env_names():
        return ["classifields", "cms", "gitlab", "reddit", "shopping", "wikipedia"]


SERVER_STATE = ServerState(
    reset_info={}
)


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. params: {url_params}")

    state = gr.State()
    state.value = SERVER_STATE

    status_text = state.value.get_status_text()

    # check if all is free
    if state.value.is_all_unavailable():
        return state, gr.Button(interactive=False), status_text
    else:
        return state, gr.Button(interactive=True), status_text


def update_reset_info(state: gr.State, env_to_reset: List[str]):
    logger.info(f"env_to_reset set to: {env_to_reset}")
    return state


def _reset_envs(env_names: List[str]):
    all_sh_to_run = []
    for env in env_names:
        sh_file = RESET_BASH_FILES[env]
        all_sh_to_run.append(f"sh {sh_file}")

    done_command = (
        f"""curl -X POST http://localhost:{SERVER_PORT}/call/done_resetting """
        """-s -H "Content-Type: application/json" -d '{"""
        """"data": ["""
        f"""{json.dumps(env_names)}"""
        """]}'"""
    )
    
    all_sh_to_run.append(done_command)
    combined_command = " && ".join(all_sh_to_run)

    with open(RESET_LOG_FILE, 'w', encoding='utf-8') as log_file:
        pass
    log_file = open(RESET_LOG_FILE, 'a', encoding='utf-8')

    logger.info(f"Running command: {combined_command}")

    _ = subprocess.Popen(
        combined_command,
        shell=True,
        start_new_session=True,
        stdin=log_file,
        stdout=log_file,
        stderr=log_file,
        text=True
    )
    return


def reset_environment(env_to_reset: List[str]):
    server_state = SERVER_STATE
    # check if all envs are free
    for env in env_to_reset:
        if not server_state.get_env_state(env) in [FREE, DOWN]:
            gr.Warning("Some environments you selected are not free/down")
            return gr.Button(interactive=True), server_state.get_status_text()

    server_state.update_env_state(env_to_reset, RESETING)
    _reset_envs(env_to_reset)
    
    server_state.reset_info = {
        "user": 'blabla'
    }
    logger.info(f"reset_environment. reset_info: {server_state.reset_info}")
    if server_state.is_all_unavailable():
        gr.Button(interactive=False), server_state.get_status_text()
    return gr.Button(interactive=True), server_state.get_status_text()


def reserve_eval(env_names: List[str]):
    server_state = SERVER_STATE

    # check if all free
    for env in env_names:
        if server_state.get_env_state(env) != FREE:
            return server_state.get_status_text()  # noop

    server_state.update_env_state(env_names, EVALUATING)
    logger.info(f"reserve_eval. status: {server_state}")
    return server_state.get_status_text()


def done_eval(env_names: List[str]):
    server_state = SERVER_STATE

    server_state.update_env_state(env_names, FREE)

    logger.info(f"done_eval. status: {server_state}")
    return server_state.get_status_text()


def done_resetting(env_names: List[str]):
    server_state = SERVER_STATE

    server_state.update_env_state(env_names, FREE)

    logger.info(f"done_resetting. status: {server_state}")
    return server_state.get_status_text()


def refresh_log():
    server_state = SERVER_STATE
    with open(RESET_LOG_FILE, "r") as f:
        log_data = f.read()
    
    check_server_down()
    return log_data, server_state.get_status_text()


def clear_log():
    with open(RESET_LOG_FILE, 'w', encoding='utf-8') as log_file:
        pass
    return ""


def check_server_down():
    for s_name, url in ENV_WEBSITE_URLS.items():
        # noop if the server is still resetting
        if SERVER_STATE.get_env_state(s_name) == RESETING:
            continue
        
        # try sending a request to see if its 200 ok. Maybe reset script had errors.
        try:
            response = requests.get(url)
            if response.status_code != 200:
                SERVER_STATE.update_env_state([s_name], DOWN)
            else:
                # if the server is was down, and is now good, update it to free
                if SERVER_STATE.get_env_state(s_name) == DOWN:
                    SERVER_STATE.update_env_state([s_name], FREE)
        except Exception as e:
            print(e)
            SERVER_STATE.update_env_state([s_name], DOWN)
    return


title_markdown = ("""
# VWA Env Management Server
""")


def build_demo(embed_mode):
    with gr.Blocks(title="VWA Env Management", theme=gr.themes.Default()) as demo:
        # dummy variable since all components here need to be gr Componetns
        # the real ones are initialized inside the demo.load() function
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        status_text = gr.HighlightedText(
            value=[("Env status", FREE)],
            combine_adjacent=False,
            show_legend=False,
            color_map={
                RESETING: "red",
                EVALUATING: "yellow",
                FREE: "green",
                DOWN: "gray"
            }
        )
        
        with gr.Column():
            log_area = gr.TextArea(
                label="Log",
                placeholder="Env reset log will be displayed here",
            )

            with gr.Row(elem_id="buttons") as _:
                with gr.Column():
                    env_to_reset = gr.CheckboxGroup(
                        choices=ServerState.get_env_names(),
                        label="reset_envs"
                    )
                    reset_btn = gr.Button(value="Reset Env", variant="primary", interactive=True)
                with gr.Column():
                    refresh_log_btn = gr.Button(value="Refresh Log", interactive=True)
                    clear_log_btn = gr.Button(value="Clear Log", interactive=True)

            # invisible buttons for apis
            update_status_btn = gr.Button(interactive=False, visible=False)
            reserve_eval_btn = gr.Button(interactive=False, visible=False)
            done_eval_btn = gr.Button(interactive=False, visible=False)
        
        url_params = gr.JSON(visible=False)

        # Register listeners
        env_to_reset.change(
            fn=update_reset_info,
            inputs=[state, env_to_reset],
            outputs=[state]
        )

        reset_btn.click(
            reset_environment,
            [env_to_reset],
            [reset_btn, status_text]
        )

        refresh_log_btn.click(
            refresh_log,
            [],
            [log_area, status_text]
        )

        clear_log_btn.click(
            clear_log,
            [],
            [log_area]
        )

        ### listenting to curl requests
        ## these will not be visible on the UI
        update_status_btn.click(
            done_resetting,
            [env_to_reset],
            [status_text]
        )

        reserve_eval_btn.click(
            reserve_eval,
            [env_to_reset],
            [status_text]
        )

        done_eval_btn.click(
            done_eval,
            [env_to_reset],
            [status_text]
        )
        

        ### runs this every refresh
        demo.load(
            load_demo,
            [url_params],
            [state, reset_btn, status_text],
            queue=False
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=55405)
    parser.add_argument("--max_threads", type=int, default=10)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    SERVER_PORT = args.port

    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(
        api_open=False
    ).launch(
        max_threads=args.max_threads,
        server_name=args.host,
        server_port=args.port,
        show_api=True,
        share=args.share
    )

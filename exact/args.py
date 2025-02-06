import os
from dataclasses import dataclass, field


@dataclass
class CommonArgs:
    max_steps: int = 15
    test_config_base_dir: str = field(
        default="data/osworld_data/evaluation_examples",
        metadata={"help": "Base directory for test configs"}
    )
    test_all_meta_path: str = field(
        default="data/osworld_data/evaluation_examples/test_tiny.json",
        metadata={"help": "Path to the test meta file"}
    )
    domain: str = "all"
    result_dir: str = field(
        default="data/osworld_data/eval_results",
        metadata={"help": "Base directory to store results (will also use exp_name, agent, model_id, etc)"}
    )
    exp_name: str = field(
        default="debug", metadata={"help": "Experiment name"}
    )
    save_agent_state: bool = field(
        default=False, metadata={"help": "Save agent state at the end. Only applicable to resumable agents"}
    )

    def __post_init__(self):
        return


@dataclass
class EnvArgs:
    path_to_vm: str | None = None
    cache_dir: str = field(
        default="cache", metadata={"help": "Cache directory used by the env to save files/images"}
    )
    headless: bool = False
    save_recording: bool = False
    action_space: str = field(
        default="pyautogui", metadata={"help": "Action type"}
    )
    observation_type: str = field(
        default="a11y_tree",
        metadata={"choices": ["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]},
    )
    screen_width: int = 1920
    screen_height: int = 1080
    sleep_after_execution: float = 1.0

    ## used by DynamicPooledDesktopEnv
    n_sim_instances: int = 0

    def __post_init__(self):
        return


@dataclass
class AgentArgs:
    agent: str = "react"
    policy: str = field(
        default="react", metadata={"help": "The policy method to be used by the agent."}
    )
    model: str = field(
        default="gpt-4o-mini", metadata={"help": "Model name or path (e.g., to load tokenizer)"}
    )
    model_id: str = field(
        default="gpt-4o-mini", metadata={"help": "Model name used when saving results"}
    )
    model_api_provider: str = field(
        default="sglang", metadata={"help": "Model provider"}
    )
    max_trajectory_length: int = field(
        default=3, metadata={"help": "Max length of past obs/action pairs (i.e., turns) used as context"}
    )
    user_prompt_prefix: str = field(
        default="",
        metadata={
            "help": "Prefix to be added to the user prompt at each turn. Inputting 'none' is same as ''.",
            "choices": ["", "none", "reasoning_v1", "explorative_v1"]
        }
    )
    ### generation
    temperature: float = 1.0
    top_p: float = 0.9
    max_tokens: int = 1500
    stop_token: str | None = None
    a11y_tree_max_tokens: int = 10000
    ### input prompt
    force_context_truncation: bool = field(
        default=False,
        metadata={
            "help": "(when enabled) IF max_context_length is short (e.g., <16k), will cut off content *inside* a turn to fit in len limit."
        }
    )
    flatten_chat_msg: bool = field(
        default=False,
        metadata={
            "help": "Whether to flatten list data in a chat turn BEFORE piping into openai api client."
        }
    )
    flatten_engine: str = field(
        default="vllm",
        metadata={
            "help": "How to flatten list data in a chat turn. vllm uses '\n' to join them, and sglang splits them as multiple user turns." ,
            "choices": ["vllm", "sglang"]
        }
    )
    max_context_length: int = field(
        default=0,
        metadata={
            "help": "max_context_length of the tokenizer. If 0, infer from model config" ,
        }
    )

    def __post_init__(self):
        _api_base = os.environ.get("POLICY_LLM_API_BASE", "")
        assert _api_base != "", "Did you forget to set your POLICY_LLM_API_BASE?"
        print(f"Using API base: {_api_base}")

        assert os.environ.get("POLICY_LLM_API_KEY", "") != "", "Did you forget to set your POLICY_LLM_API_KEY?"
        
        if self.user_prompt_prefix == "none":
            self.user_prompt_prefix = ""
        return


@dataclass
class ValueArgs:
    value_func: str = field(
        default="v_func",
        metadata={"help": "The name of the value function."}
    )
    vf_model: str = field(
        default="gpt-4o-mini",
        metadata={"help": "Model name or path (e.g., to load tokenizer)"}
    )
    vf_serve_model_name: str = field(
        default="",
        metadata={"help": "The model name used to query the LLM API. Default uses vf_model unless "}
    )
    vf_model_api_provider: str = field(
        default="sglang", metadata={"help": "Model provider"}
    )
    vf_max_trajectory_length: int = field(
        default=3, metadata={"help": "Max length of past obs/action pairs (i.e., turns) used as context"}
    )
    vf_temperature: float = 1.0
    vf_top_p: float = 0.9
    
    ### input prompt
    vf_force_context_truncation: bool = False
    vf_flatten_chat_msg: bool = False
    vf_flatten_engine: str = field(
        default="vllm",
        metadata={
            "help": "How to flatten list data in a chat turn. vllm uses '\n' to join them, and sglang splits them as multiple user turns." ,
            "choices": ["vllm", "sglang"]
        }
    )
    vf_max_context_length: int = 0

    def __post_init__(self):
        _api_base = os.environ.get("VALUE_LLM_API_BASE", "")
        assert _api_base != "", "Did you forget to set your VALUE_LLM_API_BASE?"
        print(f"Using API base: {_api_base}")

        assert os.environ.get("VALUE_LLM_API_KEY", "") != "", "Did you forget to set your VALUE_LLM_API_KEY?"
        return


@dataclass
class DummyValueArgs:
    value_func: str = "dummy"


@dataclass
class RDummyValueArgs:
    value_func: str = "rdummy"
    vf_db_path: str = None
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseAgentArguments:
    agent_type: str = field(
        default="prompt",
        metadata={"choices": ["prompt", "search_refactored"]},
    )
    prompt_constructor_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Prompt constructor type. If none, read from instruction_path."
        },
    )
    action_set_tag: str = field(
        default="id_accessibility_tree",
        metadata={"help": "Action type"},
    )
    instruction_path: str = field(
        default="src/prompts/vwa/state_action_agent.json",
    )
    provider: str = field(default="openai")
    model: str = field(default="gpt-3.5-turbo-0613")
    mode: str = field(default="chat")

    # used by tree search type of agents
    value_function_method: str = field(
        default="DirectCoTValueFunction",
        metadata={
            "help": "What value function (prompting) method to use.",
            "choices": [
                "DirectCoTValueFunction",
                "DirectCoTV2ValueFunction",
                "DirectCoTV3ValueFunction",
                "CoTwRubricValueFunction",
                "ReinforcedCoTwRubricValueFunction",
                "ReinforcedCoTwRubricValueFunctionV2",
                "CoTwDebateValueFunction",
                "ReinforcedDebateValueFunction",
                "Always0p5ValueFunction"
            ],
        },
    )


@dataclass
class BaseReinforcedAgentArguments(BaseAgentArguments):
    ## used by ReinforcedMCoTPromptConstructor
    rlm_provider: str = field(default="openai")
    rlm_model: str = field(default="gpt-3.5-turbo-0613")
    rlm_mode: str = field(default="chat")
    embedding_provider: str = field(default="openai")
    embedding_model: str = field(default="text-embedding-3-small")
    db_path: str = field(
        default="",
        metadata={"help": "Path to the database directory."},
    )
    
    # for reinforced policy functions
    max_reflections_per_task: int = field(
        default=3,
        metadata={"help": "Max reflections per task."}
    )
    reflection_threshold: float = field(
        default=0.1,
        metadata={"help": "Reflection threshold. If unexpected score is below this, no reflection will be generated."}
    )
    min_retrieval_score: float = field(
        default=0.25,
        metadata={"help": "Min retrieval score to filter irrelavant reflections durnig retrieval."}
    )
    max_to_retrieve: int = field(
        default=2,
        metadata={"help": "Max reflections to retrieve for improving policy model."}
    )
    use_gt_success: bool = field(
        default=False,
        metadata={"help": "Use ground truth success during policy reflection."}
    )

    # for reinforced value functions
    value_max_reflections_per_task: int = field(
        default=2,
        metadata={"help": "Max reflections per task."}
    )
    value_reflection_threshold: float = field(
        default=0.5,
        metadata={"help": "Reflection threshold. If unexpected score is below this, no reflection will be generated."}
    )
    value_min_retrieval_score: float = field(
        default=0.25,
        metadata={"help": "Min retrieval score to filter irrelavant reflections durnig retrieval."}
    )
    value_max_to_retrieve: int = field(
        default=1,
        metadata={"help": "Max reflections to retrieve for improving value model."}
    )
    value_use_gt_success: bool = field(
        default=False,
        metadata={"help": "Use ground truth success during value reflection."}
    )


@dataclass
class AgentArguments(BaseAgentArguments):
    agent_type: str = field(
        default="prompt",
        metadata={"choices": ["prompt", "search_refactored", "tot", "mcts"]},
    )
    instruction_path: str = field(
        default="src/prompts/vwa/state_action_agent.json",
    )
    parsing_failure_th: int = field(
        default=3,
        metadata={
            "help": "When consecutive parsing failures exceed this threshold, the agent will terminate early."
        },
    )
    repeating_action_failure_th: int = field(
        default=5,
        metadata={
            "help": "When consecutive repeated actions exceed this threshold, the agent will terminate early."
        },
    )
    captioning_model: str = field(
        default="Salesforce/blip2-flan-t5-xl",
        metadata={
            "choices": ["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
            "help": "Captioning backbone for accessibility tree alt text."
        },
    )
    ### generic lm args used by all methods
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.9)
    context_length: int = field(default=0)
    max_tokens: int = field(default=384)
    stop_token: str = field(default=None)
    max_retry: int = field(
        default=1,
        metadata={
            "help": "max retry times to perform generations when parsing fails"
        },
    )
    max_obs_length: int = field(
        default=3840,
        metadata={
            "help": "when not zero, will truncate the observation to this length before feeding to the model"
        },
    )
    ### search agent specific
    max_depth: int = field(
        default=4, metadata={"help": "Max depth for search agents."}
    )
    branching_factor: int = field(
        default=5, metadata={"help": "Branching factor at each step for the search agent."}
    )
    puct: float = field(
        default=1.0, metadata={"help": "PUCT value for the MCTS agent."}
    )
    search_algo: str = field(
        default="vf", metadata={"help": "Search algorithm to use", "choices": ["vf", "bfs", "dfs"]}
    )
    vf_budget: int = field(
        default=20, metadata={"help": "Budget for the number of value function evaluations."}
    )
    time_budget: float = field(
        default=-1.0, metadata={"help": "Time (in minutes) budget for the search agent. vf_budget will be IGNORED when this is >0."}
    )
    value_function: str = field(
        default="gpt4o", metadata={"help": "What value function to use."}
    )

    def __post_init__(self):
        if self.agent_type in ["search_refactored", "mcts"]:
            assert not (self.vf_budget is None and self.time_budget < 0.0), "Value function budget or Time budget should be specified."
        return


@dataclass
class ReinforcedAgentArguments(AgentArguments, BaseReinforcedAgentArguments):
    agent_type: str = field(
        default="prompt",
        metadata={"choices": [
            "prompt",
            "search_refactored",
            "tot",
            "mcts",
            "rmcts",
            "rmcts_mad"
        ]},
    )

    ## generic rlm args
    rlm_temperature: float = field(default=0.7)
    rlm_top_p: float = field(default=0.9)
    rlm_context_length: int = field(default=0)
    rlm_max_obs_length: int = field(
        default=512,
        metadata={
            "help": "when not zero, will truncate the observation to this length before feeding to the model"
        },
    )
    rlm_max_tokens: int = field(default=384)
    rlm_stop_token: str = field(default=None)
    rlm_max_retry: int = field(
        default=1,
        metadata={
            "help": "max retry times to perform generations when parsing fails"
        },
    )
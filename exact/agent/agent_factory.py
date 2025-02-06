from exact.args import AgentArgs, ValueArgs, DummyValueArgs, RDummyValueArgs
from exact.agent.react_dev import PromptAgent
from exact.agent.rjn_sampling import (
    RejectionSamplingAgentArgs, RejectionSamplingAgent, RejectionSamplingSearchMetadata
)
from exact.agent.best_of_n import (
    BestofNAgentArgs, BestofNAgent, BestofNAgentSearchMetadata
)
from exact.agent.mcts import (
    MCTSAgentArgs, MCTSAgent, MCTSAgentSearchMetadata
)
from exact.agent.mcts_vca import (
    MCTSwVCActionAgentArgs, MCTSwVCActionAgent
)
from exact.agent.rmcts import (
    RMCTSAgentArgs, RMCTSAgent
)
from exact.agentic.value_function import (
    CoTValueArgs, TrainedValueArgs,
    ValueFunction, CoTValueFunction, SingleDebateValueFunction, TrainedValueFunction
)
from exact.agentic.rvalue_function import (
    NoopReinforcedCoTValueArgs, NoopReinforcedCoTValueFunction, NoopReinforcedSingleDebateValueFunction
)


def get_agent_arg_cls(agent_name: str):
    if agent_name == PromptAgent.name:
        agent_arg_cls = AgentArgs
    elif agent_name == RejectionSamplingAgent.name:
        agent_arg_cls = RejectionSamplingAgentArgs
    elif agent_name == BestofNAgent.name:
        agent_arg_cls = BestofNAgentArgs
    elif agent_name == MCTSAgent.name:
        agent_arg_cls = MCTSAgentArgs
    elif agent_name == MCTSwVCActionAgent.name:
        agent_arg_cls = MCTSwVCActionAgentArgs
    elif agent_name == RMCTSAgent.name:
        agent_arg_cls = RMCTSAgentArgs
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")
    return agent_arg_cls


def construct_search_metadata(
    agent_name: str,
    # metadata used by various search based agents
    result_dir: str,
    env,
    env_args,
    task_config
):
    if agent_name == PromptAgent.name:
        search_metadata = {}
    elif agent_name == RejectionSamplingAgent.name:
        search_metadata = RejectionSamplingSearchMetadata(
            result_dir=result_dir,
            env=env,
            env_args=env_args,
            task_config=task_config
        )
    elif agent_name == BestofNAgent.name:
        search_metadata = BestofNAgentSearchMetadata(
            result_dir=result_dir,
            env=env,
            env_args=env_args,
            task_config=task_config
        )
    elif agent_name in [
        MCTSAgent.name, RMCTSAgent.name,
        MCTSwVCActionAgent.name,
    ]:
        search_metadata = MCTSAgentSearchMetadata(
            result_dir=result_dir,
            env=env,
            env_args=env_args,
            task_config=task_config
        )
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")
    return search_metadata


def construct_agent(
    agent_name: str,
    agent_args,
    action_space,
    observation_type,
    value_func: ValueFunction = None,
):
    if agent_name == PromptAgent.name:
        agent = PromptAgent(
            args=agent_args,
            action_space=action_space,
            observation_type=observation_type,
        )
    elif agent_name == RejectionSamplingAgent.name:
        agent = RejectionSamplingAgent(
            args=agent_args,
            action_space=action_space,
            observation_type=observation_type,
        )
    elif agent_name == BestofNAgent.name:
        agent = BestofNAgent(
            args=agent_args,
            value_function=value_func,
            action_space=action_space,
            observation_type=observation_type
        )
    elif agent_name == MCTSAgent.name:
        agent = MCTSAgent(
            args=agent_args,
            value_function=value_func,
            action_space=action_space,
            observation_type=observation_type
        )
    elif agent_name == MCTSwVCActionAgent.name:
        agent = MCTSwVCActionAgent(
            args=agent_args,
            action_space=action_space,
            observation_type=observation_type
        )
    elif agent_name == RMCTSAgent.name:
        agent = RMCTSAgent(
            args=agent_args,
            value_function=value_func,
            action_space=action_space,
            observation_type=observation_type
        )
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")
    return agent


def get_value_arg_cls(value_func_name: str):
    if value_func_name == ValueFunction.name:
        value_arg_cls = ValueArgs  # not really using value function
    elif value_func_name == CoTValueFunction.name:
        value_arg_cls = CoTValueArgs
    elif value_func_name == SingleDebateValueFunction.name:
        value_arg_cls = CoTValueArgs  # same args as CoT
    elif value_func_name == TrainedValueFunction.name:
        value_arg_cls = TrainedValueArgs
    elif value_func_name == NoopReinforcedCoTValueFunction.name:
        value_arg_cls = NoopReinforcedCoTValueArgs
    elif value_func_name == NoopReinforcedSingleDebateValueFunction.name:
        value_arg_cls = NoopReinforcedCoTValueArgs  # same args as NoopReinforcedCoT
    elif value_func_name == ReinforcedValueReACTPolicy.name:
        value_arg_cls = RDummyValueArgs
    else:
        raise ValueError(f"Unknown value function name: {value_func_name}")
    return value_arg_cls


def construct_value_function(
    value_func_name: str,
    value_func_args,
    observation_type,
    action_space,
):
    if value_func_name in [
        ValueFunction.name
    ]:
        value_function = None
    elif value_func_name == CoTValueFunction.name:
        value_function = CoTValueFunction(
            args=value_func_args,
            observation_type=observation_type,
            action_space=action_space
        )
    elif value_func_name == SingleDebateValueFunction.name:
        value_function = SingleDebateValueFunction(
            args=value_func_args,
            observation_type=observation_type,
            action_space=action_space
        )
    elif value_func_name == TrainedValueFunction.name:
        value_function = TrainedValueFunction(
            args=value_func_args,
            observation_type=observation_type,
            action_space=action_space
        )
    elif value_func_name == NoopReinforcedCoTValueFunction.name:
        value_function = NoopReinforcedCoTValueFunction(
            args=value_func_args,
            observation_type=observation_type,
            action_space=action_space
        )
    elif value_func_name == NoopReinforcedSingleDebateValueFunction.name:
        value_function = NoopReinforcedSingleDebateValueFunction(
            args=value_func_args,
            observation_type=observation_type,
            action_space=action_space
        )
    else:
        raise ValueError(f"Unknown value function name: {value_func_name}")
    return value_function
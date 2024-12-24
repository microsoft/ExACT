import json
import logging
from transformers import AutoTokenizer
from agent.prompts import PromptConstructor
from src.agentic.policy import (
    MCoTPolicyPConstructor_OLD,
    MCoTPolicyPConstructor,
    CoTPolicyPConstructor,
    ExploratoryCoTPolicyPConstructor
)
from src.agentic.rpolicy import (
    ReinforcedPolicyPConstructor,
    ReinforcedPolicyPConstructorPureTEXT
)
from src.llms import lm_config
from src.llms.tokenizer import Tokenizer
from src.agent.base_agent import FastAgent, PromptAgent
from src.agent.tot_agent import TreeOfThoughtAgent
from src.agent.search_agent import SearchAgent, SearchAgentRefactored
from src.agent.mcts_agent import MCTSAgent
from src.agent.rmcts_agent import RMCTSAgent, RMCTSwDBTAgent
from src.agentic.value_function import (
    DirectCoTValueFunction,
    CoTwRubricValueFunction,
    CoTwDebateValueFunction
)
from src.agentic.rvalue_function import (
    ReinforcedCoTwRubricValueFunction,
    ReinforcedDebateValueFunction,
    ReinforcedDebateValueFunctionNoREFLECTION # ablations
)
# ablations
from src.agentic.value_function import Always0p5ValueFunction
from src.agent.agent_args import BaseAgentArguments, BaseReinforcedAgentArguments
from agent.agent import TeacherForcingAgent


logger = logging.getLogger("logger")


def is_agent_search_typed(agent_type: str) -> bool:
    if "search" in agent_type:
        return True
    if "mcts" in agent_type:
        return True
    if "tot" in agent_type:
        return True
    return False


def _determine_prompt_constructor_cls(args: BaseAgentArguments) -> type[PromptConstructor]:
    # prioritize command line argument
    if args.prompt_constructor_type is not None:
        return eval(args.prompt_constructor_type)
    # otherwise read from instruction_path
    with open(args.instruction_path) as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    return eval(constructor_type)


def construct_agent(args: BaseAgentArguments, captioning_fn=None) -> FastAgent:
    llm_config = lm_config.construct_llm_config(args)

    agent: FastAgent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor_cls = _determine_prompt_constructor_cls(args)
        prompt_constructor = prompt_constructor_cls(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn
        )
    elif args.agent_type == "mcts":
        tokenizer = Tokenizer(args.provider, args.model)
        value_function_method = eval(args.value_function_method)()
        prompt_constructor_cls = _determine_prompt_constructor_cls(args)
        prompt_constructor = prompt_constructor_cls(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = MCTSAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            value_function=value_function_method,
            captioning_fn=captioning_fn
        )
    elif args.agent_type == "tot":
        tokenizer = Tokenizer(args.provider, args.model)
        value_function_method = eval(args.value_function_method)()
        prompt_constructor_cls = _determine_prompt_constructor_cls(args)
        prompt_constructor = prompt_constructor_cls(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = TreeOfThoughtAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            value_function=value_function_method,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn
        )
    elif args.agent_type == "search_refactored":  # allows custom value function on top of the base search agent implementation
        tokenizer = Tokenizer(args.provider, args.model)
        value_function_method = eval(args.value_function_method)()
        prompt_constructor_cls = _determine_prompt_constructor_cls(args)
        prompt_constructor = prompt_constructor_cls(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = SearchAgentRefactored(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            value_function=value_function_method,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn
        )
    elif args.agent_type == "search":
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor_cls = _determine_prompt_constructor_cls(args)
        prompt_constructor = prompt_constructor_cls(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = SearchAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn
        )
    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent


def construct_reinforced_agent(args: BaseReinforcedAgentArguments, captioning_fn=None) -> FastAgent:
    llm_config = lm_config.construct_llm_config(args)
    rlm_config = lm_config.construct_rlm_config(args)
    embedding_config = lm_config.LMConfig(
        provider=args.embedding_provider, model=args.embedding_model, mode="chat"
    )

    agent: FastAgent
    logger.info(f"Using {args.agent_type} agent with {args.value_function_method=} and {args.prompt_constructor_type=}.")
    if args.agent_type == "rmcts":
        assert "ReinforcedCoTwRubricValueFunction" in args.value_function_method
        assert "ReinforcedPolicyPConstructor" in args.prompt_constructor_type
        
        tokenizer = Tokenizer(args.provider, args.model)
        rlm_tokenizer = Tokenizer(args.rlm_provider, args.rlm_model)
        embedding_tokenizer = Tokenizer(args.embedding_provider, args.embedding_model)
        prompt_constructor_cls = _determine_prompt_constructor_cls(args)
        prompt_constructor: ReinforcedPolicyPConstructor = prompt_constructor_cls(
            args.instruction_path,
            db_path=args.db_path,
            lm_config=llm_config,
            rlm_config=rlm_config,
            embedding_config=embedding_config,
            tokenizer=tokenizer,
            rlm_tokenizer=rlm_tokenizer,
            embedding_tokenizer=embedding_tokenizer,
            # behavioral args
            max_reflections_per_task=args.max_reflections_per_task,
            reflection_threshold=args.reflection_threshold,
            min_retrieval_score=args.min_retrieval_score,
            max_to_retrieve=args.max_to_retrieve,
            use_gt_success=args.use_gt_success,
        )
        value_function_method = eval(args.value_function_method)(
            db_path=args.db_path,
            rlm_config=rlm_config,
            embedding_config=embedding_config,
            rlm_tokenizer=rlm_tokenizer,
            embedding_tokenizer=embedding_tokenizer,
            max_reflections_per_task=args.value_max_reflections_per_task,
            reflection_threshold=args.value_reflection_threshold,
            min_retrieval_score=args.value_min_retrieval_score,
            max_to_retrieve=args.value_max_to_retrieve,
            use_gt_success=args.value_use_gt_success,
        )
        agent = RMCTSAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            value_function=value_function_method,
            captioning_fn=captioning_fn
        )
    elif args.agent_type == "rmcts_mad":
        assert "ReinforcedDebateValueFunction" in args.value_function_method
        assert "ReinforcedPolicyPConstructor" in args.prompt_constructor_type

        tokenizer = Tokenizer(args.provider, args.model)
        rlm_tokenizer = Tokenizer(args.rlm_provider, args.rlm_model)
        embedding_tokenizer = Tokenizer(args.embedding_provider, args.embedding_model)
        prompt_constructor_cls = _determine_prompt_constructor_cls(args)
        prompt_constructor: ReinforcedPolicyPConstructor = prompt_constructor_cls(
            args.instruction_path,
            db_path=args.db_path,
            lm_config=llm_config,
            rlm_config=rlm_config,
            embedding_config=embedding_config,
            tokenizer=tokenizer,
            rlm_tokenizer=rlm_tokenizer,
            embedding_tokenizer=embedding_tokenizer,
            # behavioral args
            max_reflections_per_task=args.max_reflections_per_task,
            reflection_threshold=args.reflection_threshold,
            min_retrieval_score=args.min_retrieval_score,
            max_to_retrieve=args.max_to_retrieve,
            use_gt_success=args.use_gt_success,
        )
        value_function_method = eval(args.value_function_method)(
            db_path=args.db_path,
            rlm_config=rlm_config,
            embedding_config=embedding_config,
            rlm_tokenizer=rlm_tokenizer,
            embedding_tokenizer=embedding_tokenizer,
            max_reflections_per_task=args.value_max_reflections_per_task,
            reflection_threshold=args.value_reflection_threshold,
            min_retrieval_score=args.value_min_retrieval_score,
            max_to_retrieve=args.value_max_to_retrieve,
            use_gt_success=args.value_use_gt_success,
        )
        agent = RMCTSwDBTAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            value_function=value_function_method,
            captioning_fn=captioning_fn
        )
    else:
        # fall back to the simpler construct_agent
        agent = construct_agent(args, captioning_fn)
    return agent

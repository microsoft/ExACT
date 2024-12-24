#!/bin/bash
export PYTHONPATH=$(pwd)
source <path_to_api_key_envs>/.keys
export DATASET=visualwebarena
# export DATASET=webarena

export CLASSIFIEDS="<your_classifieds_domain>:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"  # Default reset token for classifieds site, change if you edited its docker-compose.yml
export SHOPPING="<your_shopping_site_domain>:7770"
export REDDIT="<your_reddit_domain>:9999"
export WIKIPEDIA="<your_wikipedia_domain>:8888"
export SHOPPING_ADMIN="<your_e_commerce_cms_domain>:7780/admin"
export GITLAB="<your_gitlab_domain>:8023"
export MAP="<your_map_domain>:3000"
export HOMEPAGE="<your_homepage_domain>:4399"


## Define the model, result directory, and instruction path variables
[[[API_PROVIDER_ENV_VARS]]]  # replaced by runners/eval/eval_vwa_parallel.py
EVAL_GPU_IDX=1

model="gpt-4o"  # "gpt-4o-mini"
model_id="gpt-4o"
instruction_path="src/prompts/vwa/jsons/p_som_cot_id_actree_3s_final.json"  # ablation with rmcts v7.2 prompts
test_config_dir="configs/visualwebarena/test_classifieds_v2"  # see runners/eval/fix_task_intents.py

# change this to "prompt" to run the baseline without search
agent="mcts"

##### start of search config
max_depth=4  # max_depth=4 means 5 step lookahead
max_steps=5
branching_factor=5  # default 5
vf_budget=20        # default 20
time_budget=5.0     # 5.0 min per step (soft maximum), will override vf_budget if > 0.0

prompt_constructor_type=MCoTPolicyPConstructor

# vfunc config
v_func_method=CoTwRubricValueFunction  # default DirectCoTValueFunction
##### end of search config

test_idx="[[[test_idx]]]"  # replaced by runners/eval/eval_vwa_parallel.py

RUN_FILE=runners/eval/eval_vwa_agent.py
SAVE_ROOT_DIR="[[[SAVE_ROOT_DIR]]]"  # replaced by runners/eval/eval_vwa_parallel.py
echo "SAVEDIR=${SAVE_ROOT_DIR}"
mkdir -p $SAVE_ROOT_DIR
cp "$0" "${SAVE_ROOT_DIR}/run.sh"

export DEBUG=True  # export DEBUG=''
####### start eval
# reset, reserving, and freeing is handled by an external script

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} \
python $RUN_FILE \
--instruction_path $instruction_path \
--test_idx $test_idx \
--model $model \
--provider $PROVIDER \
--agent_type $agent \
--prompt_constructor_type $prompt_constructor_type \
--branching_factor $branching_factor --vf_budget $vf_budget --time_budget $time_budget \
--value_function $model \
--value_function_method $v_func_method \
--result_dir $SAVE_ROOT_DIR \
--test_config_base_dir $test_config_dir \
--repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
--action_set_tag som  --observation_type image_som \
--top_p 0.95   --temperature 1.0  --max_steps $max_steps

####### end eval
python runners/utils/repartition_log_files.py $SAVE_ROOT_DIR/log_files
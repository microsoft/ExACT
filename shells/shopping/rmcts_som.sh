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
rlm_model="gpt-4o"
embedding_model="text-embedding-3-small"
instruction_path="src/prompts/vwa/jsons/p_som_cot_id_actree_3s_final.json"  # see src/prompts/vwa/to_json.py
test_config_dir="configs/visualwebarena/test_shopping_v2"  # see runners/eval/fix_task_intents.py

# change this to "prompt" to run the baseline without search
agent="rmcts"

##### start of search config
max_depth=4  # max_depth=4 means 5 step lookahead
max_steps=5
branching_factor=5  # default 5
vf_budget=20        # default 20
time_budget=5.0     # 5.0 min per step (soft maximum), will override vf_budget if > 0.0

# policy config
prompt_constructor_type=ReinforcedPolicyPConstructor  # default ReinforcedPolicyPConstructor
max_reflections_per_task=3         # default 3
reflection_threshold=0.5           # default 0.1
puct=1.0                           # default 1.0

# vfunc config
v_func_method=ReinforcedCoTwRubricValueFunction  # default ReinforcedCoTwRubricValueFunction
value_max_reflections_per_task=1   # default 2
value_reflection_threshold=0.5     # default 0.5

##### end of search config

test_idx="[[[test_idx]]]"  # replaced by runners/eval/eval_vwa_parallel.py


RUN_FILE=runners/eval/eval_vwa_ragent.py
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
--puct $puct \
--branching_factor $branching_factor --vf_budget $vf_budget --time_budget $time_budget \
--max_reflections_per_task $max_reflections_per_task \
--reflection_threshold $reflection_threshold \
--value_function $model \
--value_function_method $v_func_method \
--value_max_reflections_per_task $value_max_reflections_per_task \
--value_reflection_threshold $value_reflection_threshold \
--rlm_model $rlm_model \
--rlm_provider $PROVIDER \
--embedding_provider $EMBEDDING_MODEL_PROVIDER \
--embedding_model $embedding_model \
--db_path $SAVE_ROOT_DIR/db \
--result_dir $SAVE_ROOT_DIR \
--test_config_base_dir $test_config_dir \
--repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
--action_set_tag som  --observation_type image_som \
--top_p 0.95 --temperature 1.0 --max_steps $max_steps

##### cleanups
python runners/utils/repartition_log_files.py $SAVE_ROOT_DIR/log_files
export POLICY_LLM_API_BASE="https://api.openai.com/v1"
export POLICY_LLM_API_KEY=$OPENAI_API_KEY
export POLICY_LLM_API_VERSION=""
export POLICY_LLM_TOKEN_PROVIDER_BASE=""

export VALUE_LLM_API_BASE="https://api.openai.com/v1"
export VALUE_LLM_API_KEY=$OPENAI_API_KEY
export VALUE_LLM_API_VERSION=""
export VALUE_LLM_TOKEN_PROVIDER_BASE=""

export REFLECTION_LLM_API_BASE="https://api.openai.com/v1"
export REFLECTION_LLM_API_KEY=$OPENAI_API_KEY
export REFLECTION_LLM_API_VERSION=""
export REFLECTION_LLM_TOKEN_PROVIDER_BASE=""

# you can use openai,azure, or sglang (for locally hosted models)
# dont't forget to change the env variables above accordingly
model_api_provider=openai
vf_model_api_provider=openai
rlm_api_provider=openai

export DOCKER_DISK_SIZE="8G"
export OSWORLD_DATA_DIR=/abs_path_preferred/ExACT/data/osworld_data
export DOCKER_RUN_PRIVILEGED=true

TEST_FPATH=data/osworld_data/evaluation_examples/test_just_one.json
TEST_NAME=reprod-run1

agent=rmcts

#### policy
model=gpt-4o
model_id=gpt-4o

max_trajectory_length=3
temperature=1.0
max_steps=15
n_nodes=5  # tree size. for test run, choose a small n.
n_sim_instances=4
branching_factor=2
branching_algo=sample
prior_temperature=5.0
adv_after_n_nodes=10  # 7
adv_counter=search_itr  # subtree_size
c_func=constant
cpuct_end=1.0


embedding_model=text-embedding-3-small
embedding_api_provider=openai

rlm_model=gpt-4o
rlm_temperature=0.7
rlm_top_p=0.9

selection_metric=unexpected_and_absq  # unexpected_score
max_reflections_per_task=3
reflection_threshold=0.25
min_retrieval_score=0.7
max_to_retrieve=2
use_gt_success=False

#### value
value_func=noop_r_sad_value_func
vf_model=gpt-4o
vf_n=10
vf_temperature=0.7
vf_top_p=0.9

#### save shell script
action_space=pyautogui
observation_type=a11y_tree # change this to run in other modalities!
result_base_dir=${OSWORLD_DATA_DIR}/eval_results
SAVE_ROOT_DIR="${result_base_dir}/${agent}/${TEST_NAME}__${model_id}__${action_space}__${observation_type}"
echo "SAVEDIR=${SAVE_ROOT_DIR}"
mkdir -p $SAVE_ROOT_DIR
cp "$0" "${SAVE_ROOT_DIR}/run.sh"

#### run!
cache_root_dir="/tmp/${agent}_${model_id}_${TEST_NAME}_cache_$(date +%Y%m%d%H%M%S)"

python runners/test/run_rsearch.py \
--headless \
--test_all_meta_path $TEST_FPATH \
--model $model \
--model_id $model_id \
--model_api_provider $model_api_provider \
--agent $agent \
--max_steps $max_steps \
--n_nodes $n_nodes \
--n_sim_instances $n_sim_instances \
--branching_factor $branching_factor \
--branching_algo $branching_algo \
--prior_temperature $prior_temperature \
--adv_after_n_nodes $adv_after_n_nodes \
--adv_counter $adv_counter \
--c_func $c_func \
--cpuct_end $cpuct_end \
--observation_type $observation_type \
--temperature $temperature \
--embedding_model $embedding_model \
--embedding_api_provider $embedding_api_provider \
--rlm_model $rlm_model \
--rlm_api_provider $rlm_api_provider \
--rlm_temperature $rlm_temperature \
--rlm_top_p $rlm_top_p \
--selection_metric $selection_metric \
--max_reflections_per_task $max_reflections_per_task \
--reflection_threshold $reflection_threshold \
--min_retrieval_score $min_retrieval_score \
--max_to_retrieve $max_to_retrieve \
--use_gt_success $use_gt_success \
--value_func $value_func \
--vf_model $vf_model \
--vf_model_api_provider $vf_model_api_provider \
--vf_n $vf_n \
--vf_temperature $vf_temperature \
--vf_top_p $vf_top_p \
--cache_dir $cache_root_dir \
--exp_name $TEST_NAME
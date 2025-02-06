source [[[API_KEY_FPATH]]]
export OSWORLD_DATA_DIR=[[[OSWORLD_DATA_DIR]]]

[[[API_PROVIDER_ENV_VARS]]]

TEST_FPATH=[[[TEST_FPATH]]]
TEST_NAME=[[[TEST_NAME]]]


export DOCKER_DISK_SIZE="8G"
export DOCKER_RUN_PRIVILEGED=true
agent=rmcts

#### policy
model=gpt-4o
model_id=gpt-4o

max_trajectory_length=3
temperature=1.0
max_steps=15
n_nodes=60
n_sim_instances=30
branching_factor=10
branching_algo=random
prior_temperature=5.0
adv_after_n_nodes=20  # 7 (maybe bfactor * 2)
adv_counter=search_itr  # subtree_size
bfactor_func=exp_decay
bfactor_func_coeff=1.0
c_func=cosine
cpuct_end=0.1


embedding_model=text-embedding-3-small
embedding_api_provider=openai

rlm_model=gpt-4o
rlm_temperature=0.7
rlm_top_p=0.9

selection_metric=unexpected_and_absq  # unexpected_score
max_reflections_per_task=4
reflection_threshold=0.25
min_retrieval_score=0.5
max_to_retrieve=2
use_gt_success=False

#### value
value_func=noop_r_sad_value_func
vf_model=gpt-4o
vf_n=10
vf_temperature=0.7
vf_top_p=0.9

#### save shell script
GIT_HASH=$(git rev-parse HEAD)
action_space=pyautogui
observation_type=a11y_tree
result_base_dir=${OSWORLD_DATA_DIR}/eval_results
SAVE_ROOT_DIR="${result_base_dir}/${agent}/${TEST_NAME}__${model_id}__${action_space}__${observation_type}"
echo "SAVEDIR=${SAVE_ROOT_DIR}"
mkdir -p $SAVE_ROOT_DIR
cp "$0" "${SAVE_ROOT_DIR}/run.sh"
echo "[$(date)] running [$SAVE_ROOT_DIR/run.sh] at git=[$GIT_HASH]" >> "${SAVE_ROOT_DIR}/git_hash.txt"

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
--bfactor_func $bfactor_func \
--bfactor_func_coeff $bfactor_func_coeff \
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
export POLICY_LLM_API_BASE="https://api.openai.com/v1"
export POLICY_LLM_API_KEY=$OPENAI_API_KEY
export POLICY_LLM_API_VERSION=""
export POLICY_LLM_TOKEN_PROVIDER_BASE=""

export VALUE_LLM_API_BASE="https://api.openai.com/v1"
export VALUE_LLM_API_KEY=$OPENAI_API_KEY
export VALUE_LLM_API_VERSION=""
export VALUE_LLM_TOKEN_PROVIDER_BASE=""

# you can use openai,azure, or sglang (for locally hosted models)
# dont't forget to change the env variables above accordingly
model_api_provider=openai

export OSWORLD_DATA_DIR=/abs_path_preferred/ExACT/data/osworld_data
export DOCKER_DISK_SIZE="8G"

TEST_FPATH=data/osworld_data/evaluation_examples/test_just_one.json
TEST_NAME=reprod-run1

temperature=1.0
max_trajectory_length=3

agent=react
model=gpt-4o
model_id=gpt-4o

cache_root_dir="/tmp/${agent}_${model_id}_${TEST_NAME}_cache_$(date +%Y%m%d%H%M%S)"

action_space=pyautogui
observation_type=a11y_tree # change this to run in other modalities!
SAVE_ROOT_DIR="data/osworld_data/eval_results/${agent}/${TEST_NAME}__${model_id}__${action_space}__${observation_type}"
echo "SAVEDIR=${SAVE_ROOT_DIR}"
mkdir -p $SAVE_ROOT_DIR
cp "$0" "${SAVE_ROOT_DIR}/run.sh"

#### run!
python runners/test/run_normal.py \
--headless \
--test_all_meta_path $TEST_FPATH \
--model $model \
--model_id $model_id \
--model_api_provider $model_api_provider \
--max_steps 15 \
--max_trajectory_length $max_trajectory_length \
--agent $agent \
--observation_type $observation_type \
--temperature $temperature \
--cache_dir $cache_root_dir \
--exp_name $TEST_NAME
# ExACT: Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning

<!-- [[Website]](https://agent-e3.github.io/rmcts-exploratory-learning/) -->
<!-- [[arXiv]](https://arxiv.org/abs/2410.02052) -->

We present R-MCTS and Exploratory Learning for building o1-like models for agentic applications. Our **R-MCTS agent** extends traditional MCTS by 1) incorporating contrastive reflection, allowing agents to learn from past interactions and dynamically improve their search efficiency; and 2) using multi-agent debate to provide reliable state evaluation.

<!-- <img src="media/rmcts-simplified.gif" alt=""> -->

**Exploratory Learning** is a novel learning strategy that trains the models to explore the environment, evaluate a state, and backtrack to viable ones when it detects that the current state cannot lead to success. Our GPT-4o powered R-MCTS agent creates SOTA performance on VisualWebArena. Notably, R-MCTS and Exploratory Learning demonstrate the compute scaling properties in both training and testing time.

<!-- <img style="aspect-ratio: 3.5;" src="media/learning-data.gif"> -->

> You are currently at the branch that contains code for running OSWorld tasks. For VisualWebArena tasks, please switch to the other branch.

# Setup

1. Install the `OSWorld` repo to setup testing environment. To ensure reproducility, please use our forked version of `OSWorld` repo:
   ```bash
    # Clone the our forked version of OSWorld repository
    https://github.com/Agent-E3/OSWorld

    # Change directory into the cloned repository
    cd OSWorld

    # Optional: Create a Conda environment for OSWorld
    conda create -n osworld python=3.10.12
    conda activate osworld

    # Install required dependencies
    pip install -r requirements.txt
    pip install -e .
   ```
   Note that we will default to using `docker` as the environment for testing.
2. Clone the current repository, and install it with `pip install -e`
3. This completes the software setup. To setup data folders, configure your data folder looks like this:
    ```bash
    ExACT
    ├── data
    │   ├── osworld_data # folders such as evaluation_samples are symlinked from the OSWorld repo
    │   │   ├── docker_vm_data # stores the docker image used by OSWorld
    │   │   ├── eval_results
    │   │   └── evaluation_samples
    │   └── visualwebarena # see the VWA branch of this repo!
    │       └── eval_results
    ...
    ```
    You should take a note of the absolute path of `data/osworld_data/docker_vm_data`. This path will be used in the next step.

To ensure everything is setup correctly, you can run the following test:
```bash
# NOTE: docker cannot mount if any part of the $OSWORLD_DATA_DIR path is not 777 permission
export OSWORLD_DATA_DIR=/abs_path_preferred/ExACT/data/osworld_data
python test.py
```

# Quickstart

We provide quickstart scripts where you can run the agents for a few tasks, visualize the output, and evaluate the performance.

```bash
# export the environment variables from step 2 in SETUP section
export OSWORLD_DATA_DIR=/abs_path_preferred/ExACT/data/osworld_data
source <your_openai_api_key_envs.key>
shells/rmcts_gpt-4o.sh
```

This will, by default:
- run tasks under `TEST_FPATH` from OSWorld using R-MCTS agent
- save results to `data/osworld_data/eval_results`, including intermediate tree search visualizations


## Parallization

We provide scripts and commands to evaluate 369 tasks from OSWorld in parallel. We provide two ways to do this: perform parallel runs inside one machine (this section), or perform distributed parallel runs across multiple machines (next section).


To perform parallel runs (with both single/multiple machines), first configure the `runners/configs/*.yaml` file to certain key directorys inside each machine:
- the `runners/configs/files.yaml` should contain the following:
    ```yaml
    apple:
        api_key_fpath: <apple's path to openai_api_key_envs.key>
        osworld_data_dir: /abs_path_preferred/ExACT/data/osworld_data

    banana:
        api_key_fpath: <apple's path to openai_api_key_envs.key>
        osworld_data_dir: /abs_path_preferred/ExACT/data/osworld_data
    ```
- the `runners/configs/providers.yaml` should contain the following:
    ```yaml
    openai:
        provider: openai
        api_base: https://api.openai.com/v1
        api_key: null  # runs os.environ['OPENAI_API_KEY']
        api_version: ''
        token_provider_base: ''
    
    # if you use azure, you can add the following:
    azure:
        provider: azure
        api_base: <azure's api_base>
        api_key: empty
        api_version: <azure's api_version>
        token_provider_base: <optional, azure's token_provider_base>
    ```

Then, to perform parallel runs in a single machine (e.g., in `apple`), you can run the following command:
```bash
~/exact$ source <your_openai_api_key_envs.key>
~/exact$ python runners/parallel_runner.py \
--eval_script shells/rmcts_gpt-4o_base.sh \
--test_name <exp-run-name> \
--num_parallel 2 \
--main_api_providers openai,azure \
--run_tfiles data/osworld_data/evaluation_examples/<test_file_1.json>,data/osworld_data/evaluation_examples/<test_file_2.json> \
--machine_name apple
```


## Distributed Inf + Parallization

To perform distributed parallel runs across multiple machines, the idea is to use one machine as the manager/master, and other machines as workers. The manager will distribute the tasks and shell scripts to the workers, and the workers will run the tasks.

1. start manager in the **main machine** (e.g., `apple`)
    ```bash
    python distributed/manager.py
    ```
    which by default runs on port `12000`

2. start worker in each of your **worker machine** (e.g., in `banana`), and register it to the manager
    ```bash
    source <your_api_key_envs.key>
    export DOCKER_RUN_PRIVILEGED=true # any machine specific configurations
    python distributed/worker.py \
    --mname banana \
    --addr_for_manager banana.cs.your-uni.com \
    --port 12038 \
    --manager_api apple.cs.your-uni.com :12000
    ```

3. configure what tasks to run in every worker machine you registered in a yaml file, such as `distributed/jobs/test.yaml`
    ```yaml
    banana:
        num_parallel: 2
        main_api_providers:
            - openai
            - azure
        run_tfiles: # path inside banana
            - data/osworld_data/evaluation_examples/<test_file_1.json>
            - data/osworld_data/evaluation_examples/<test_file_2.json>
    ```
3. submit job in the **main machine** to the worker, with tasks specified in the `distributed/jobs/test.yaml` file
    ```bash
    python distributed/dist_cli.py \
    --mode dist_run_parallel_runner \
    --manager_addr localhost:12000 \
    --base_shell_fpath shells/rmcts_gpt-4o_base.sh \
    --test_name <exp-run-name> \
    --dist_config_path distributed/jobs/test.yaml
    ```
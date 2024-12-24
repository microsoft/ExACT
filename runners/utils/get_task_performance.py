import json
import os
import re
import glob
import pandas as pd
import argparse


ERROR_KEY_WORD = "is No match found"


def get_error_df(env_dir, is_v1=False):
    task_error_count = []
    for html_file in glob.glob(env_dir + "/*.html"):
        with open(html_file, "r") as file:
            data = file.read()
            error_count = data.count(ERROR_KEY_WORD)
            task_error_count.append((os.path.basename(html_file), error_count))
    return pd.DataFrame(task_error_count, columns=["task", "error_count"])


def _sum_llm_tokens(llm_token_dict: dict):
    # llm_token_dict contains {model_name: {prompt_tokens: int, completion_tokens: int}}
    all_prompt_tokens = 0
    all_completion_tokens = 0
    for _, tokens in llm_token_dict.items():
        all_prompt_tokens += tokens["prompt_tokens"]
        all_completion_tokens += tokens["completion_tokens"]
    return all_prompt_tokens, all_completion_tokens


def _extract_domain_from_dir(env_dir):
    if 'classifields' in env_dir or 'classifieds' in env_dir:
        return "configs/visualwebarena/test_classifieds_v2"
    elif 'shopping' in env_dir:
        return "configs/visualwebarena/test_shopping_v2"
    elif 'reddit' in env_dir:
        return "configs/visualwebarena/test_reddit_v2"
    elif 'gitlab' in env_dir:
        return "configs/webarena/test_gitlab_v2"
    else:
        raise ValueError(f"Unknown domain for {env_dir}")


def get_task_perf(env_dir):
    perf_folder = os.path.join(env_dir, "performances")
    config_folder = _extract_domain_from_dir(env_dir)

    output = []
    tokens_not_found_tids = []
    for perf_json in glob.glob(perf_folder + "/*.json"):
        # tid = re.search(r"performance_(\d+)_(\d+).json", perf_json).group(1)
        with open(perf_json, "r") as file:
            data = json.load(file)

            task_configs = data["eval_configs"]
            scores = data["scores"]
            times = data["times"]
            if isinstance(task_configs, str):
                task_configs = [task_configs]
                scores = [scores]
                times = [times]

            llm_tokens = data.get("llm_tokens", [{}] * len(scores))  # old versions don't have this field
            if isinstance(llm_tokens, dict):
                llm_tokens = [llm_tokens] * len(scores)
            
            for task, score, time, llm_token in zip(task_configs, scores, times, llm_tokens):
                task_id = re.search(r"(\d+).json", task).group(1)
                # get task difficulty
                config_file = os.path.join(config_folder, f"{task_id}.json")
                with open(config_file, "r") as fread:
                    config_data = json.load(fread)
                task_difficulty = config_data.get("overall_difficulty", "unknown")
                
                # process llm tokens abit
                prompt_tokens, completion_tokens = _sum_llm_tokens(llm_token)
                if len(llm_token) == 0:
                    tokens_not_found_tids.append(task_id)
                
                output.append([task_id, task_difficulty, score, time, prompt_tokens, completion_tokens])

    # try to check log files to find llm_tokens if not found
    if len(tokens_not_found_tids) > 0:
        print(f"Tokens not found for {len(tokens_not_found_tids)} tasks: {tokens_not_found_tids}")
    return pd.DataFrame(output, columns=["task", "difficulty", "score", "time", "prompt_tokens", "completion_tokens"])


if __name__ == "__main__":
    # e.g., python get_task_performance.py data/visualwebarena/eval_results/cot_som/InternVL2-Llama3-76B_v4.1_1s_shopping_0-100
    parser = argparse.ArgumentParser()
    parser.add_argument("rundir", type=str)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ### check other ones
    stats = {}
    error_df = get_error_df(args.rundir)
    speed_perf_df = get_task_perf(args.rundir)

    stats["error_count_per_task"] = error_df['error_count'].mean()
    stats["hash_error_per_task"] = (error_df['error_count'] > 0).mean()
    stats["mean_score"] = speed_perf_df['score'].mean()
    stats["mean_time"] = speed_perf_df['time'].mean()
    # group score by difficulty
    grouped = speed_perf_df.groupby('difficulty')
    for name in ['easy', 'medium', 'hard']:
        if name in grouped.groups:
            stats[name] = grouped.get_group(name)['score'].mean()
        else:
            stats[name] = 0
    # skip the zero tokens as these may be bugged
    filtered_speed_df = speed_perf_df[(speed_perf_df['prompt_tokens'] > 0)]  # backward compatibility for older run scripts
    if len(filtered_speed_df) != len(speed_perf_df):
        print(f"NOTE: Token estimation is based on {len(filtered_speed_df)} tasks")
    stats['mean_prompt_token'] = filtered_speed_df['prompt_tokens'].mean()
    stats['mean_completion_token'] = filtered_speed_df['completion_tokens'].mean()

    print(json.dumps(stats))

    if args.verbose:
        print('num tasks:', len(speed_perf_df))
        # print incorrect tids
        all_incorrect_df = speed_perf_df[speed_perf_df['score'] == 0]
        print(f"Incorrect tasks: {len(all_incorrect_df)}")
        all_tids = sorted(all_incorrect_df['task'].values.tolist(), key=lambda x: int(x))
        print(','.join(all_tids))
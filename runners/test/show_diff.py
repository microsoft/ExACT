import os
import numpy as np
import argparse
import json
import pandas as pd


def _sum_tokens(token_perf_list: dict):
    # assume
    total_prompt_tokens = sum([v['prompt_tokens'] for v in token_perf_list.values()])
    total_completion_tokens = sum([v['completion_tokens'] for v in token_perf_list.values()])
    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
    }


def get_result(target_dir: str):
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None
    
    all_result_for_analysis = []
    for domain in os.listdir(target_dir):
        if 'logs' in domain or domain.endswith(".json"):
            continue
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        
        ## compute score per domain
        for example_id in os.listdir(domain_path):
            example_path = os.path.join(domain_path, example_id)
            if os.path.isdir(example_path):
                if "performance.json" in os.listdir(example_path):
                    perf_file = os.path.join(example_path, "performance.json")
                    try:
                        with open(perf_file, "r") as fread:
                            result = json.load(fread)
                    except Exception as e:
                        print(f"Error in reading {perf_file}: {e}")
                        continue
                    
                    score = float(result["score"])
                    success = 1 if score == 1.0 else 0
                    time_spent = float(result["time_spent"])
                    token_usage = _sum_tokens(result["llm_token"])

                    all_result_for_analysis.append({
                        "domain": domain,
                        "example_id": example_id,
                        "score": score,
                        "success": success,
                        "time": time_spent,
                        "ptokens (k)": token_usage['total_prompt_tokens']/1000,
                    })
    return all_result_for_analysis


def get_diff(result_dir_1: str, result_dir_2: str):
    dir_1_result = get_result(result_dir_1)
    dir_2_result = get_result(result_dir_2)

    dir_1_result_df = pd.DataFrame(dir_1_result)
    dir_2_result_df = pd.DataFrame(dir_2_result)

    dir_1_result_df.set_index(["domain", "example_id"], inplace=True)
    dir_2_result_df.set_index(["domain", "example_id"], inplace=True)

    ### print same performances
    shared_df = dir_1_result_df.join(
        dir_2_result_df,
        how="inner",
        lsuffix="_1",
        rsuffix="_2",
    )
    same_perf_df = shared_df[shared_df["score_1"] == shared_df["score_2"]]
    print('>'*10)
    if args.show_same:
        print(f"Same performances:")
        print(same_perf_df[["score_1", "score_2", "time_1", "time_2", "ptokens (k)_1", "ptokens (k)_2"]])
    else:
        print(f'Same performances: {len(same_perf_df)} examples')

    diff_perf_df = shared_df[shared_df["score_1"] != shared_df["score_2"]]
    print('>'*10)
    print(f"Different performances:")
    print(diff_perf_df[["score_1", "score_2", "time_1", "time_2", "ptokens (k)_1", "ptokens (k)_2"]])

    print('>'*10)
    # not in both
    not_in_1 = dir_2_result_df.index.difference(dir_1_result_df.index)
    not_in_2 = dir_1_result_df.index.difference(dir_2_result_df.index)
    if args.show_missing:
        print(f"Examples missing in dir_1:")
        for domain, example_id in not_in_1:
            print(f"{domain:<10s}{example_id}")

        print()
        print(f"Examples missing in dir_2: {len(not_in_2)}")
        for domain, example_id in not_in_2:
            print(f"{domain:<10s}{example_id}")
    else:
        print(f"Examples missing in dir_1: {len(not_in_1)}")
        print(f"Examples missing in dir_2: {len(not_in_2)}")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_1", type=str)
    parser.add_argument("dir_2", type=str)
    parser.add_argument("--show_same", action="store_true")
    parser.add_argument("--show_missing", action="store_true")
    args = parser.parse_args()

    get_diff(args.dir_1, args.dir_2)

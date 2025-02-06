import os
import pandas as pd
import numpy as np
import argparse
import json
import traceback
import pickle
import lzma
import os
import concurrent.futures

os.environ['OSWORLD_DATA_DIR'] = "/tmp"   # placeholder so that imports would work
from exact.agent.mcts import Node


SUBTASK_MAPPNIG = {
    "office": ["libreoffice_calc", "libreoffice_impress", "libreoffice_writer"],
    "daily": ["vlc", "thunderbird", "chrome"],
    "professional": ["gimp", "vs_code"],
    "os": ["os"],
    "workflow": ["multi_apps"],
}


def _sum_tokens(token_perf_list: dict):
    # assume
    total_prompt_tokens = sum([v['prompt_tokens'] for v in token_perf_list.values()])
    total_completion_tokens = sum([v['completion_tokens'] for v in token_perf_list.values()])
    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
    }


def _get_nested_list(domain_keys: list[str], metric: str, domain_result: dict):
    output = []
    for key in domain_keys:
        output += domain_result.get(key, {}).get(metric, [])
    return output


def _get_substask_performance(subtask_name, domain_result: dict):
    all_scores = _get_nested_list(
        SUBTASK_MAPPNIG[subtask_name], 'score', domain_result
    )
    all_ptokens = _get_nested_list(
        SUBTASK_MAPPNIG[subtask_name], 'prompt_tokens', domain_result
    )
    all_ctokens = _get_nested_list(
        SUBTASK_MAPPNIG[subtask_name], 'completion_tokens', domain_result
    )
    return {
        "score": all_scores,
        "success": [1 if s == 1.0 else 0 for s in all_scores],
        "prompt_tokens": all_ptokens,
        "completion_tokens": all_ctokens,
    }


def _get_tree_stats_if_exist(task_dir: str):
    root_fpath = os.path.join(task_dir, "search_tree.pkl.xz")
    if not os.path.exists(root_fpath):
        return False, {
            "real_nodes": 0,
            "leaf_nodes": 0,
            "max_depth": 0,
            "one_branch_success": None,
        }
    
    with lzma.open(root_fpath, "rb") as fread:
        root_node: Node = pickle.load(fread)
    
    all_nodes = root_node._get_all_children()

    real_nodes = 0
    leaf_nodes = 0
    max_depth = 0
    for resp, node in all_nodes:
        if not node._need_simluation_n_eval:
            real_nodes += 1
            ## leaf node if no children
            ## or no children is evaled
            if not node.children:
                leaf_nodes += 1
            elif node.is_terminal:
                leaf_nodes += 1
            else:
                no_eval_children = all([
                    child._need_simluation_n_eval
                    for child in node.children.values()
                ])
                if no_eval_children:
                    leaf_nodes += 1
            
            max_depth = max(max_depth, node.depth)
    
    search_meta_info_fpath = os.path.join(task_dir, "search_meta_info", "performance.json")
    if os.path.exists(search_meta_info_fpath):
        with open(search_meta_info_fpath, "r") as fread:
            search_meta_info = json.load(fread)
        one_branch_success = search_meta_info['score']
    else:
        one_branch_success = None
    return True, {
        "real_nodes": real_nodes,
        "leaf_nodes": leaf_nodes,
        "max_depth": max_depth,
        "one_branch_success": one_branch_success,
    }


def _get_depth_stats_if_exist(task_dir: str):
    depth_result_fpath = os.path.join(task_dir, "performance_gt_depth.json")
    if not os.path.exists(depth_result_fpath):
        return False, {}

    with open(depth_result_fpath, "r") as fread:
        depth_result = json.load(fread)
    
    if 'all_scores' in depth_result:
        ### bon, has many branches. Each branch has a score per depth
        all_scores = depth_result['all_scores']  # a 2D list
        # calculate success per depth
        depth_success = {}
        max_depth = 15  # default in osworld
        for d_i in range(max_depth):
            score_at_depth = [s[d_i] if len(s) > d_i else s[-1] for s in all_scores]
            success_at_depth = 1 if any([s == 1.0 for s in score_at_depth]) else 0
            depth_success[d_i+1] = success_at_depth
        return True, depth_success
    else:
        ### react like, only one trajectory per task
        scores = depth_result['score']  # a 1D list
        # calculate success per depth
        depth_success = {}
        max_depth = 15  # default in osworld
        for d_i in range(max_depth):
            if d_i >= len(scores):
                depth_success[d_i+1] = scores[-1]
            else:
                success_at_depth = 1 if scores[d_i] == 1.0 else 0
                depth_success[d_i+1] = success_at_depth
        return True, depth_success
    return


def _read_single_task_perf(
    domain: str, example_id: str, example_path: str,
    domain_result: dict, all_result_for_analysis: dict,
    _all_score: list, _all_success: list, _all_time: list
):
    perf_file = os.path.join(example_path, "performance.json")
    try:
        with open(perf_file, "r") as fread:
            result = json.load(fread)
    except Exception as e:
        print(f"Error in reading {perf_file}: {e}")
        return

    if domain not in domain_result:
        domain_result[domain] = {
            'score': [],
            'success': [],
            'prompt_tokens': [],
            'completion_tokens': [],
            'tree_stats': [],
            'depth_stats': [],
        }
        all_result_for_analysis[domain] = {}
    
    score = float(result["score"])
    success = 1 if score == 1.0 else 0
    time_spent = float(result["time_spent"])
    token_usage = _sum_tokens(result["llm_token"])
    found_tree, tree_stats = _get_tree_stats_if_exist(example_path)
    found_depth, depth_success_stats = _get_depth_stats_if_exist(example_path)

    domain_result[domain]['score'].append(score)
    domain_result[domain]['success'].append(success)
    domain_result[domain]['prompt_tokens'].append(token_usage['total_prompt_tokens'])
    domain_result[domain]['completion_tokens'].append(token_usage['total_completion_tokens'])
    if found_tree:
        domain_result[domain]['tree_stats'].append(tree_stats)
    if found_depth:
        domain_result[domain]['depth_stats'].append(depth_success_stats)

    all_result_for_analysis[domain][example_id] = {'score': score, 'success': success, 'time': time_spent}
    _all_score.append(score)
    _all_success.append(success)
    _all_time.append(time_spent)
    return


def get_result(target_dir: str):
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None
    
    _all_score = []
    _all_success = []
    _all_time = []

    domain_result = {}
    all_result_for_analysis = {}

    for domain in os.listdir(target_dir):
        if 'logs' in domain or domain.endswith(".json"):
            continue
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        
        ## compute score per domain
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "performance.json" in os.listdir(example_path):
                        # works since order doesn't matter, we just need summary stats
                        future = executor.submit(
                            _read_single_task_perf,
                            domain, example_id, example_path,
                            domain_result, all_result_for_analysis,
                            _all_score, _all_success, _all_time
                        )
                        futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                future.result()


    for domain in domain_result:
        domain_perf = np.mean(domain_result[domain]['success'])
        domain_score = np.mean(domain_result[domain]['score'])
        ptokens = np.mean(domain_result[domain]['prompt_tokens']) / 1000
        ctokens = np.mean(domain_result[domain]['completion_tokens']) / 1000
        n_tasks = len(domain_result[domain]['success'])

        n_trees_found = len(domain_result[domain]['tree_stats'])
        if n_trees_found > 0:
            n_leaf_nodes = np.mean([v['leaf_nodes'] for v in domain_result[domain]['tree_stats']])
            n_nodes = np.mean([v['real_nodes'] for v in domain_result[domain]['tree_stats']])
            max_depth = np.mean([v['max_depth'] for v in domain_result[domain]['tree_stats']])
            max_depth_std = np.std([v['max_depth'] for v in domain_result[domain]['tree_stats']])
            one_branch_success = [v['one_branch_success'] for v in domain_result[domain]['tree_stats'] if v['one_branch_success'] is not None]
            one_branch_score = np.mean(one_branch_success)
            print((
                f"{domain: <25s} success={domain_perf:8.2%} out of {n_tasks:3d} tasks;\t"
                f"score={domain_score:8.4f}\t\t"
                f"ptokens={ptokens:8.2f}k\t"
                f"ctokens={ctokens:8.2f}k\t"
                f"trees={n_trees_found:3d}; "
                f"nodes={n_nodes:6.2f}; "
                f"leaf_nodes={n_leaf_nodes:6.2f}; "
                f"max_depth={max_depth:6.2f}(pm{max_depth_std:4.2f});\t"
                f"gt_score={one_branch_score:8.4f} from {len(one_branch_success):3d} tasks"
            ))
        else:
            print((
                f"{domain: <25s} success={domain_perf:8.2%} out of {n_tasks:3d} tasks;\t"
                f"score={domain_score:8.4f}\t\t"
                f"ptokens={ptokens:8.2f}k\t"
                f"ctokens={ctokens:8.2f}k"
            ))
    ### depth analysis
    domain_depth_success = []
    domain_names = []
    for domain in domain_result:
        n_depth_found = len(domain_result[domain]['depth_stats'])
        if n_depth_found > 0:
            success_per_depth_df = pd.DataFrame(domain_result[domain]['depth_stats'])
            avg_success_per_depth = success_per_depth_df.mean()
            domain_depth_success.append(avg_success_per_depth.to_dict())
            domain_names.append(f"{domain} (n={n_depth_found})")
    if len(domain_depth_success) > 0:
        print(">>>>>>>>>>>>>")
        domain_depth_success_df = pd.DataFrame(domain_depth_success, index=domain_names)
        pd.options.display.float_format = "{:8.2%}".format
        print(domain_depth_success_df)

    ### the following runs only if you ran at least one task for each subtask
    try:
        print(">>>>>>>>>>>>>")
        office_data = _get_substask_performance('office', domain_result)
        daily_data = _get_substask_performance('daily', domain_result)
        professional_data = _get_substask_performance('professional', domain_result)
        os_data = _get_substask_performance('os', domain_result)
        workflow_data = _get_substask_performance('workflow', domain_result)

        # print
        office_perf = np.mean(office_data['success'])
        office_score = np.mean(office_data['score'])
        daily_perf = np.mean(daily_data['success'])
        daily_score = np.mean(daily_data['score'])
        professional_perf = np.mean(professional_data['success'])
        professional_score = np.mean(professional_data['score'])
        os_perf = np.mean(os_data['success'])
        os_score = np.mean(os_data['score'])
        workflow_perf = np.mean(workflow_data['success'])
        workflow_score = np.mean(workflow_data['score'])
        print((
            f"{'Office': <25s} success={office_perf:8.2%}\t"
            f"{'score':>21s}={office_score:8.4f}\t"
            f"{'ptokens':>15s}={np.mean(office_data['prompt_tokens'])/1000:8.2f}k\t"
            f"ctokens={np.mean(office_data['completion_tokens'])/1000:8.2f}k"
        ))
        print((
            f"{'Daily': <25s} success={daily_perf:8.2%}\t"
            f"{'score':>21s}={daily_score:8.4f}\t"
            f"{'ptokens':>15s}={np.mean(daily_data['prompt_tokens'])/1000:8.2f}k\t"
            f"ctokens={np.mean(daily_data['completion_tokens'])/1000:8.2f}k"
        ))
        print((
            f"{'Professional': <25s} success={professional_perf:8.2%}\t"
            f"{'score':>21s}={professional_score:8.4f}\t"
            f"{'ptokens':>15s}={np.mean(professional_data['prompt_tokens'])/1000:8.2f}k\t"
            f"ctokens={np.mean(professional_data['completion_tokens'])/1000:8.2f}k"
        ))
        print((
            f"{'OS': <25s} success={os_perf:8.2%}\t"
            f"{'score':>21s}={os_score:8.4f}\t"
            f"{'ptokens':>15s}={np.mean(os_data['prompt_tokens'])/1000:8.2f}k\t"
            f"ctokens={np.mean(os_data['completion_tokens'])/1000:8.2f}k"
        ))
        print((
            f"{'Workflow': <25s} success={workflow_perf:8.2%}\t"
            f"{'score':>21s}={workflow_score:8.4f}\t"
            f"{'ptokens':>15s}={np.mean(workflow_data['prompt_tokens'])/1000:8.2f}k\t"
            f"ctokens={np.mean(workflow_data['completion_tokens'])/1000:8.2f}k"
        ))
        
        all_score = np.mean(_all_score)
        all_success = np.mean(_all_success)
        all_time = np.mean(_all_time)
        overall = {
            "office": {"score": office_perf, "n_task": len(office_data)},
            "daily": {"score": daily_perf, "n_task": len(daily_data)},
            "professional": {"score": professional_perf, "n_task": len(professional_data)},
            "os": {"score": os_perf, "n_task": len(os_data)},
            "workflow": {"score": workflow_perf, "n_task": len(workflow_data)},
        }
        with open(os.path.join(target_dir, "all_result.json"), "w") as fwrite:
            json.dump({
                'n_task': len(_all_score),
                'score': all_score,
                'success': all_success,
                'time': all_time,
                "overall": overall,
                "details": all_result_for_analysis,
            }, fwrite, indent=4, sort_keys=True)
        
        if not _all_score:
            print("New experiment, no result yet.")
            return None
        else:
            print(">>>>>>>>>>>>>")
            print(f"Runned:{len(_all_score):3d}{'':>12s}Score: {all_score:.2%}{'':>12s}Total Time spent: {np.sum(_all_time):.2f}min")
            return _all_score
    except Exception as e:
        print(traceback.format_exc())
        print(f"Error in computing subtask performance: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=str)
    args = parser.parse_args()

    get_result(args.target_dir)

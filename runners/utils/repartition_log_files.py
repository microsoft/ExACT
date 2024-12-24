import argparse
import os
import glob
import re


def partition_log_file(lines: list[str]):
    found_files = {}
    lines_to_flush = []
    prev_filename = ""
    for line in lines:
        if "[Config file]:" in line:
            # parse the file id from 2024-08-27 05:23:26,449] INFO@logger [/workspace/generalized_tree_search/gts-dev/runners/eval/eval_vwa_searchagent_v6.py:225] [Config file]: /tmp/tmpcd9hgubm/15.json
            matched_task_id = re.search(r"Config file\]: .+/(\d+)\.json", line)
            task_id = matched_task_id.group(1)
            new_filename = f"task_{task_id}.log.txt"

            if prev_filename == "":
                # first log file, skip
                prev_filename = new_filename
            elif prev_filename != new_filename:
                # flush
                found_files[prev_filename] = lines_to_flush
                lines_to_flush = []
                prev_filename = new_filename

        lines_to_flush.append(line)
    
    # flush the last file
    if len(lines_to_flush) > 0:
        found_files[prev_filename] = lines_to_flush
    return found_files


def partition_log_dir(log_dir: str):
    all_log_files = glob.glob(f"{log_dir}/log*.txt")
    # sort by file creation time, so that later runs overwrite earlier runs
    all_log_files.sort(key=os.path.getctime)
    for log_file in all_log_files:
        with open(log_file, 'r') as fread:
            lines = fread.readlines()
        repartitioned_files = partition_log_file(lines)

        for filename, content in repartitioned_files.items():
            if filename == "":
                continue
            out_file_name = os.path.join(log_dir, filename)
            with open(out_file_name, 'w') as fwrite:
                fwrite.writelines(content)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Repartition log files')
    parser.add_argument('log_dir', type=str, help='Directory containing log files')
    args = parser.parse_args()

    partition_log_dir(args.log_dir)
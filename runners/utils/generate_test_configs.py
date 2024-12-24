"""Make sure your environment variables are all set before running this script"""
import json
import os

from browser_env.env_config import *


def main() -> None:
    DATASET = os.environ["DATASET"]
    if DATASET == "webarena":
        print("DATASET: webarena")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        print(f"SHOPPING_ADMIN: {SHOPPING_ADMIN}")
        print(f"GITLAB: {GITLAB}")
        print(f"WIKIPEDIA: {WIKIPEDIA}")
        print(f"MAP: {MAP}")
        
        # inp_paths = ["configs/webarena/test_webarena.raw.json"]
        inp_paths = [
            "configs/webarena/test_gitlab_v2.raw.json",  # only gitlab had some ambiguous intents, others look fine
            "configs/webarena/test_map.raw.json",
        ]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
            "__SHOPPING_ADMIN__": SHOPPING_ADMIN,
            "__GITLAB__": GITLAB,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__MAP__": MAP,
        }
    elif DATASET == "visualwebarena":
        print("DATASET: visualwebarena")
        print(f"CLASSIFIEDS: {CLASSIFIEDS}")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        # v1
        # inp_paths = [
        #     "configs/visualwebarena/test_classifieds.raw.json",
        #     "configs/visualwebarena/test_shopping.raw.json",
        #     "configs/visualwebarena/test_reddit.raw.json",
        # ]
        # v2
        inp_paths = [
            "configs/visualwebarena/test_classifieds_v2.raw.json",
            "configs/visualwebarena/test_shopping_v2.raw.json",
            "configs/visualwebarena/test_reddit_v2.raw.json",
        ]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__CLASSIFIEDS__": CLASSIFIEDS,
        }
    else:
        raise ValueError(f"Dataset not implemented: {DATASET}")
        
    for inp_path in inp_paths:
        output_dir = inp_path.replace('.raw.json', '')
        os.makedirs(output_dir, exist_ok=True)
        with open(inp_path, "r") as f:
            raw = f.read()
        for k, v in replace_map.items():
            raw = raw.replace(k, v)

        with open(inp_path.replace(".raw", ""), "w") as f:
            f.write(raw)
        data = json.loads(raw)
        for idx, item in enumerate(data):
            with open(os.path.join(output_dir, f"{idx}.json"), "w") as f:
                json.dump(item, f, indent=2)
    return


if __name__ == "__main__":
    main()
import glob
import importlib
import json
import os


# use the current directory as the root
def run() -> None:
    """Convert all python files in agent/prompts to json files in agent/prompts/jsons

    Python files are easiser to edit
    """
    for p_file in glob.glob(f"src/prompts/vwa/raw/*.py"):
        # import the file as a module
        base_name = os.path.basename(p_file).replace(".py", "")
        module = importlib.import_module(f"src.prompts.vwa.raw.{base_name}")
        prompt = module.prompt
        # save the prompt as a json file
        os.makedirs("src/prompts/vwa/jsons", exist_ok=True)
        with open(f"src/prompts/vwa/jsons/{base_name}.json", "w+") as f:
            json.dump(prompt, f, indent=2)
    print(f"Done converting python prompt files to json prompts")


if __name__ == "__main__":
    run()

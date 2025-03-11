import base64
import json
import logging
import re
import xml.etree.ElementTree as ET
import tiktoken
import os
import tempfile
from mm_agents.accessibility_tree_wrap.heuristic_retrieve import filter_nodes, draw_bounding_boxes
from PIL import Image
from io import BytesIO


logger = logging.getLogger("src.env")


attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"


def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')


def encoded_img_to_pil_img(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    return image


def resize_image_from_bytes(image_bytes, size=(960, 540)):
    image = Image.open(BytesIO(image_bytes))
    resized = image.resize(size)

    buffered = BytesIO()
    resized.save(buffered, format="PNG")
    return buffered.getvalue()


def save_to_tmp_img_file(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    tmp_img_path = os.path.join(tempfile.mkdtemp(), "tmp_img.png")
    image.save(tmp_img_path)

    return tmp_img_path


def linearize_accessibility_tree(accessibility_tree, platform="ubuntu"):

    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = ["tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"]

    # Linearize the accessibility tree nodes into a table format
    for node in filtered_nodes:
        if node.text:
            text = (
                node.text if '"' not in node.text \
                    else '"{:}"'.format(node.text.replace('"', '""'))
            )

        elif node.get("{{{:}}}class".format(class_ns_windows), "").endswith("EditWrapper") \
                and node.get("{{{:}}}value".format(_value_ns)):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = (node_text if '"' not in node_text \
                        else '"{:}"'.format(node_text.replace('"', '""'))
                    )
        else:
            text = '""'

        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag, node.get("name", ""),
                text,
                node.get("{{{:}}}class".format(_attributes_ns), "") if platform == "ubuntu" else node.get("{{{:}}}class".format(class_ns_windows), ""),
                node.get("{{{:}}}description".format(_attributes_ns), ""),
                node.get('{{{:}}}screencoord'.format(_component_ns), ""),
                node.get('{{{:}}}size'.format(_component_ns), "")
            )
        )

    return "\n".join(linearized_accessibility_tree)


def tag_screenshot(screenshot, accessibility_tree, platform="ubuntu"):
    nodes = filter_nodes(ET.fromstring(accessibility_tree), platform=platform, check_image=True)
    # Make tag screenshot
    marks, drew_nodes, element_list, tagged_screenshot = draw_bounding_boxes(nodes, screenshot)

    return marks, drew_nodes, tagged_screenshot, element_list


def parse_actions_from_string(input_string):
    if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
        return [input_string.strip()]
    # Search for a JSON string within the input string
    actions = []
    matches = re.findall(r'```json\s+(.*?)\s+```', input_string, re.DOTALL)
    if matches:
        # Assuming there's only one match, parse the JSON string into a dictionary
        try:
            for match in matches:
                action_dict = json.loads(match)
                actions.append(action_dict)
            return actions
        except json.JSONDecodeError as e:
            return f"Failed to parse JSON: {e}"
    else:
        matches = re.findall(r'```\s+(.*?)\s+```', input_string, re.DOTALL)
        if matches:
            # Assuming there's only one match, parse the JSON string into a dictionary
            try:
                for match in matches:
                    action_dict = json.loads(match)
                    actions.append(action_dict)
                return actions
            except json.JSONDecodeError as e:
                return f"Failed to parse JSON: {e}"
        else:
            try:
                action_dict = json.loads(input_string)
                return [action_dict]
            except json.JSONDecodeError:
                raise ValueError("Invalid response format: " + input_string)


def parse_code_from_string(input_string):
    input_string = "\n".join([line.strip() for line in input_string.split(';') if line.strip()])
    if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
        return [input_string.strip()]

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"
    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)

    # The regex above captures the content inside the triple backticks.
    # The `re.DOTALL` flag allows the dot `.` to match newline characters as well,
    # so the code inside backticks can span multiple lines.

    # matches now contains all the captured code snippets

    codes = []

    for match in matches:
        match = match.strip()
        commands = ['WAIT', 'DONE', 'FAIL']  # fixme: updates this part when we have more commands

        if match in commands:
            codes.append(match.strip())
        elif match.split('\n')[-1] in commands:
            if len(match.split('\n')) > 1:
                codes.append("\n".join(match.split('\n')[:-1]))
            codes.append(match.split('\n')[-1])
        else:
            codes.append(match)

    return codes


def parse_code_from_som_string(input_string, masks):
    # parse the output string by masks
    tag_vars = ""
    for i, mask in enumerate(masks):
        x, y, w, h = mask
        tag_vars += "tag_" + str(i + 1) + "=" + "({}, {})".format(int(x + w // 2), int(y + h // 2))
        tag_vars += "\n"

    actions = parse_code_from_string(input_string)

    for i, action in enumerate(actions):
        if action.strip() in ['WAIT', 'DONE', 'FAIL']:
            pass
        else:
            action = tag_vars + action
            actions[i] = action

    return actions


def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(linearized_accessibility_tree)
    if len(tokens) > max_tokens:
        linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
        linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree


def _post_process_obs(obs: dict, observation_type: str, metadata: dict):
    """returns raw obs from environment to input obs for the agent
    """
    platform = metadata['platform']
    a11y_tree_max_tokens = metadata['a11y_tree_max_tokens']

    if observation_type in ["screenshot", "screenshot_a11y_tree"]:
        base64_image = encode_image(obs["screenshot"])
        linearized_accessibility_tree = linearize_accessibility_tree(
            accessibility_tree=obs["accessibility_tree"],
            platform=platform
        ) if observation_type == "screenshot_a11y_tree" else None

        # logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

        if linearized_accessibility_tree:
            linearized_accessibility_tree = trim_accessibility_tree(
                linearized_accessibility_tree,
                a11y_tree_max_tokens
            )

        if observation_type == "screenshot_a11y_tree":
            return{
                "screenshot": base64_image,
                "accessibility_tree": linearized_accessibility_tree
            }
        else:
            return {
                "screenshot": base64_image,
                "accessibility_tree": None
            }
    elif observation_type == "a11y_tree":
        linearized_accessibility_tree = linearize_accessibility_tree(
            accessibility_tree=obs["accessibility_tree"],
            platform=platform
        )
        # logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

        if linearized_accessibility_tree:
            linearized_accessibility_tree = trim_accessibility_tree(
                linearized_accessibility_tree,
                a11y_tree_max_tokens
            )
        return {
            "screenshot": None,
            "accessibility_tree": linearized_accessibility_tree
        }
    elif observation_type == "som":
        # Add som to the screenshot
        masks, drew_nodes, tagged_screenshot, linearized_accessibility_tree = tag_screenshot(
            obs["screenshot"],
            obs["accessibility_tree"],
            platform
        )
        base64_image = encode_image(tagged_screenshot)
        # logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

        if linearized_accessibility_tree:
            linearized_accessibility_tree = trim_accessibility_tree(
                linearized_accessibility_tree,
                a11y_tree_max_tokens
            )
        return {
            "screenshot": base64_image,
            "accessibility_tree": linearized_accessibility_tree
        }
    else:
        raise ValueError("Invalid observation_type type: " + observation_type)


class ObsPostProcessor:
    def __init__(self, observation_type: str, platform: str, a11y_tree_max_tokens: int) -> None:
        self.observation_type = observation_type
        self.platform = platform
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        return

    def __call__(self, obs: dict) -> dict:
        metadata = {
            "platform": self.platform,
            "a11y_tree_max_tokens": self.a11y_tree_max_tokens
        }
        return _post_process_obs(obs, self.observation_type, metadata)
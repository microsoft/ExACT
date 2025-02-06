VFUNC_COT_PYAUTOGUI_INTRO = f"""
You are an expert in evaluating the performance of a computer agent. The agent is designed to help a human user navigate a website to complete a task.
Given the user's intent, the agent's action history, the final observation (i.e., state) of the computer, your goal is to decide whether the agent's execution completed the task correctly.
If the current states/actions haven't achieved the task but the agent seems to be on the right track, you should indicate this as well.

Note that the VALID actions the agent can take are as follows:
- can only interact with the computer through the `pyautogui` library, which allows it to perform mouse clicks, keyboard inputs, and other GUI interactions. Its action will be formatted as a Python code block.
- can return special codes like `WAIT`, `FAIL`, or `DONE` to indicate waiting for some time, task is impossible to complete by design, or completed the task, respectively.
- NO OTHER ACTIONS ARE ALLOWED except for the ones mentioned above.

Below is the user's intent, and the last few observation/action pairs of the agent's attempt to solve the provided task.
""".strip()


__VFUNC_COMMON_GUIDELINES = """
*IMPORTANT*
Note that the agent's proposed actions may NOT contain the correct coordinates, and may NOT be executed correctly in the environment (e.g., requiring pop-ups or dialogs to be clicked first).
You should base your evaluation on the OBSERVATION(s) instead of the proposed actions.

To evaluate task progress, choose one of the following options and provide your thought process.
EVALUATION:
A. The agent's last action/current observation fulfilled all requirements in the user's intent, and no further actions is needed. If the task is infeasible by design, the agent indicated this by returning ```FAIL```.
B. The agent is very close to finishing the task. Only one more action (e.g., return ```DONE``` or ```FAIL```) is needed to complete the task.
C. You are certain the agent is on the right track to success, but given the current observation, it needs more than one action to complete the task.
D. The agent may be on the right track, but the proposed actions were not executed correctly/matched expectation. We should try other actions to make better progress.
E. The agent's last action/current observation does NOT make progress towards the task completion (assuming it is feasible). The agent is NOT on the right track to success.
Note that *correctly* identifying a task as infeasible by design is as valid/important as correctly completing a feasible task (i.e., option A).
However, if the task is feasible but the agent identified it as infeasible, this should be penalized (i.e., option E).
""".strip()


VFUNC_COT_FINAL_PROMPT = f"""
Now, it is your turn to evaluate the progress of the agent's execution so far.

{__VFUNC_COMMON_GUIDELINES}

Format your response into three lines as shown below. Keep your response CONCISE.
Checks: <based on the user's intent, come up with NO MORE THAN THREE heuristic checks to determine if the task is completed correctly>
Thoughts: <your thoughts and reasoning process to evaluate the agent's performance>
EVALUATION: <one of A, B, C, D, or E>
""".strip()


VFUNC_SINGLE_DEBATE_FINAL_PROMPT = f"""
Now, it is your turn to evaluate the progress of the agent's execution so far.

{__VFUNC_COMMON_GUIDELINES}

Format your response into four lines as shown below. Keep your response CONCISE.
Reasons why the last action was/was not executed correctly based on the last observation: <your reasonings>
Reasons indicating positive progress toward task completion (if you think it is feasible) based on past observations: <your reasonings>
Reasons to try other actions, or why continuing from this state may lead to incorrect results: <your reasonings>
EVALUATION: <one of A, B, C, D, or E>
""".strip()


VFUNC_TRAINED_FINAL_PROMPT = f"""
Now, it is your turn to evaluate the progress of the agent's execution so far.

{__VFUNC_COMMON_GUIDELINES}
""".strip()
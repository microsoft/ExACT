RPOLICY_COT_ADDITIONAL_INTRO = """
**To complete this task correctly, it is crucial to adhere to the following guidelines:**
1. You should *try* to complete the task in as few actions as possible. However, making slow and steady progress is better than making hasty mistakes.
2. You should only issue actions that exists/are VALID given the OBSERVATIONS. Otherwise, there may be unintended consequences.
3. Most tasks are solvable within 10 steps. If you are convinced the task is infeasible by design (unlikely but possible), remember to return ```FAIL``` to indicate this.
""".strip()


RPOLICY_COT_ADDITIONAL_INTRO_W_REFL = """
**To complete this task correctly, it is crucial to adhere to the following guidelines:**
1. You should *try* to complete the task in as few actions as possible. However, making slow and steady progress is better than making hasty mistakes.
2. You should only issue actions that exists/are VALID given the OBSERVATIONS. Otherwise, there may be unintended consequences.
3. You should use insights from REFLECTIONS, if applicable, to better plan the next action for the current task.
4. Most tasks are solvable within 10 steps. If you are convinced the task is infeasible by design (unlikely but possible), remember to return ```FAIL``` to indicate this.
""".strip()


RPOLICY_COT_EXPECTATION_PROMPT = """
What do you expect to happen after taking this action? Briefly describe what you think will appear in the NEXT OBSERVATION after performing the action.
DO NOT propose any specific actions/answers that you will take next.
""".strip()


RPOLICY_COT_REFLECTION_PROMPT = """
Is this observation what you expected? If not, what mistakes made by the agent might explain the mismatch between the action and expected outcome? You should consider, **IF APPLICABLE**:
- (control) whether the proposed action coordinates exist/are valid given the observations.
- (control) whether the proposed actions correctly triggered the corresponding buttons/elements (e.g., clicking the top-left position of a button/element MAY NOT trigger it; if there are pop-ups/dialogs that need to be clicked first; or if double-clicking (`doubleClick`) should be used instead of a single click).
- (planning) whether the agent is making any incorrect assumptions about the task or the environment (e.g., while unlikely, it is possible that the task is inherently infeasible).
- (planning) whether the agent prematurely returned `DONE`, such as forgetting to do/fulfill certain part of the task or simply forgetting to save the changes made.
- and etc.
If you faced the same situation again, what would you do differently?

Recall that the OBJECTIVE was [{instruction}]. According to our evaluation, the agent have {task_status_str} the OBJECTIVE after executing a few more steps from this state.
DO NOT propose any specific actions, but only MOST IMPORTANT and INFORMATIVE suggestions for future agents to improve their actions. Keep your response WITHIN 100 words.
""".strip()
def _add_modality_key_for_sglang_messages(messages: list):
    for turn in messages:
        turn_content = turn["content"]
        if isinstance(turn_content, list):
            for parts in turn_content:
                if parts['type'] == "image_url":
                    parts["modalities"] = "multi-images"  # sglang needs this
        # noop if its just string
    return messages
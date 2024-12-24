import tiktoken
from transformers import AutoTokenizer


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        self.provider = provider
        if provider in ["openai", "azure"]:
            # for o1, use gpt-4o tokenizer
            if "o1" in model_name:
                self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
            else:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
        elif provider in ["huggingface", "sglang"]:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                # add_special_tokens=False,
                add_bos_token=False,
                add_eos_token=False,
            )
        elif provider == "google":
            self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
        else:
            raise NotImplementedError
        return

    def encode(self, text: str) -> list[int]:
        if self.provider in ["huggingface", "sglang"]:
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)
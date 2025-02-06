import tiktoken
from transformers import AutoTokenizer, AutoConfig


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str, max_context_length: int = 0) -> None:
        self.provider = provider
        if provider in ["openai", "azure"]:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
            self.max_context_length = max_context_length or 128_000
        elif provider in ["huggingface", "sglang"]:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                add_bos_token=False,
                add_eos_token=False,
            )
            # model_max_length = self.tokenizer.model_max_length  ## this is NOT reliable
            config = AutoConfig.from_pretrained(model_name)
            model_max_length = getattr(config, "max_position_embeddings", None)
            if model_max_length is None or model_max_length > 1e8:
                print('Unable to get model_max_length, using 32k as default')
                model_max_length = max_context_length or 32_000
            self.max_context_length = max_context_length or model_max_length
        elif provider == "google":
            self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
            self.max_context_length = max_context_length or 128_000
        else:
            raise NotImplementedError
        print(f"Model {model_name} max length set to: {self.max_context_length}")
        return

    def encode(self, text: str) -> list[int]:
        if self.provider in ["huggingface", "sglang"]:
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)
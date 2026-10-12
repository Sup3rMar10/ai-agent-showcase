"""Chat backends. The agent only needs `chat(messages) -> str`, so models are interchangeable and tests can script replies."""


class LocalChatModel:
    """Greedy decoding with a local Hugging Face chat model (CPU unless CUDA is available)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 120):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except TypeError:  # older transformers: torch_dtype
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model = model.to(self.device).eval()

    def chat(self, messages: list[dict]) -> str:
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None,
                                      top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class ScriptedLLM:
    """Returns pre-written replies in order. Lets tests exercise the agent loop with no model."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> str:
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0) if self.replies else '{"action": "final", "answer": "(script exhausted)"}'

import gc
from typing import List, Dict, Optional, Any
from ..contracts.llm import LLM
from ..utils.io import read_text


class LlamaCppClient(LLM):
    """
    LLM Client implementation utilizing `llama-cpp-python` for local GGUF model execution.
    Designed for memory efficiency and configurable GPU layer offloading.
    """

    def __init__(self, model_name: str, backend: Dict[str, Any], prompt_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the LlamaCppClient configuration parameters.
        """
        self.model_file = backend.get("model_file", model_name)
        self.n_ctx = backend.get("n_ctx", 8192)
        self.n_gpu_layers = backend.get("n_gpu_layers", -1)
        self.backend = backend
        self.default_params = params or {}
        self._model: Any | None = None
        self.prompt_template: str | None = None
        
        if prompt_path:
            self.prompt_template = read_text(prompt_path)

    def _load_resources(self) -> Any:
        """
        Lazy loads the Llama model instance into memory upon first invocation.
        """
        if self._model is None:
            from llama_cpp import Llama

            b_kwargs = {k: v for k, v in self.backend.items() if k not in ["type", "model_file", "n_ctx", "n_gpu_layers"]}
            self._model = Llama(
                model_path=self.model_file,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                **b_kwargs
            )
        return self._model

    def generate(self, template_kwargs: Dict[str, Any], system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """
        Generates text completion for a single prompt using chat-formatted execution.
        """
        messages = []
        prompt = self.prompt_template.format(**template_kwargs) if self.prompt_template else ""
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Executes a multi-turn chat sequence using native Llama.cpp chat formatting.
        """
        model = self._load_resources()
        gen_kwargs = {**self.default_params, **kwargs}
        temp = gen_kwargs.pop("temperature", 0.0)
        max_tok = gen_kwargs.pop("max_tokens", gen_kwargs.pop("max_new_tokens", 512))

        response = model.create_chat_completion(
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
            **gen_kwargs
        )
        return response["choices"][0]["message"]["content"] or ""

    def generate_batch(self, template_kwargs_list: List[Dict[str, Any]], system_prompt: Optional[str] = None, **kwargs: Any) -> List[str]:
        """
        Sequentially generates completions for a list of prompts.
        """
        return [self.generate(tk, system_prompt=system_prompt, **kwargs) for tk in template_kwargs_list]

    def release_resources(self) -> None:
        """
        Unloads the GGUF model from VRAM/RAM and invokes garbage collection.
        """
        if self._model is not None:
            del self._model
            self._model = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
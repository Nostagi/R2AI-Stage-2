import gc
from typing import List, Dict, Optional, Any
from ..contracts.llm import LLM
from ..utils.io import read_text


class VllmClient(LLM):
    """
    LLM Client implementation utilizing `vLLM` for high-throughput, GPU-accelerated inference.
    Features native continuous batching and advanced VRAM management for sequential pipelines.
    """
    _model_cache: Dict[str, Any] = {}
    _ref_counts: Dict[str, int] = {}

    def __init__(self, model_name: str, backend: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the VllmClient with engine configurations and default sampling parameters.
        """
        self.model_path = model_name
        self.gpu_memory_utilization = backend.get("gpu_memory_utilization", 0.85)
        self.max_model_len = backend.get("max_model_len", 8192)
        self.dtype = backend.get("dtype", "auto")
        self.quantization = backend.get("quantization", None)
        self.backend = backend
        self.default_params = params or {}

        cache_key = self._get_cache_key()
        cls = self.__class__
        cls._ref_counts[cache_key] = cls._ref_counts.get(cache_key, 0) + 1

    def _get_cache_key(self) -> str:
        return f"{self.model_path}_{self.gpu_memory_utilization}_{self.max_model_len}_{self.dtype}_{self.quantization}"

    @property
    def _model(self) -> Any:
        return self._model_cache.get(self._get_cache_key())

    def _ensure_model_downloaded(self) -> None:
        hf_repo = self.backend.get("hf_repo")
        if hf_repo and self.model_path.startswith("./"):
            import os
            if not os.path.exists(self.model_path):
                print(f"Downloading {hf_repo} to {self.model_path}...")
                from huggingface_hub import snapshot_download
                from src.config import get_settings
                token = get_settings().hf_token
                snapshot_download(repo_id=hf_repo, local_dir=self.model_path, token=token)

    def _load_resources(self) -> Any:
        """
        Lazy loads the vLLM Engine into memory, allocating the specified VRAM portion.
        """
        self._ensure_model_downloaded()
        cache_key = self._get_cache_key()

        if cache_key not in self._model_cache:
            from vllm import LLM

            b_kwargs = {k: v for k, v in self.backend.items() if k not in ["type", "gpu_memory_utilization", "max_model_len", "dtype", "quantization"]}
            self._model_cache[cache_key] = LLM(
                model=self.model_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                quantization=self.quantization,
                trust_remote_code=True,
                **b_kwargs
            )
        return self._model

    def _get_sampling_params(self, **kwargs: Any) -> Any:
        """
        Constructs the vLLM SamplingParams object using instance defaults or overrides.
        """
        from vllm import SamplingParams

        gen_kwargs = {**self.default_params, **kwargs}
        temp = gen_kwargs.pop("temperature", 0.1)
        top_p = gen_kwargs.pop("top_p", 0.95)
        max_tok = gen_kwargs.pop("max_tokens", gen_kwargs.pop("max_new_tokens", 1024))

        return SamplingParams(
            temperature=temp,
            top_p=top_p,
            max_tokens=max_tok,
            **gen_kwargs
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Executes a multi-turn chat sequence by applying the model's tokenizer chat template.
        """
        model = self._load_resources()
        tokenizer = model.get_tokenizer()
        
        # Apply the appropriate chat template (e.g., Qwen/Llama3 formats)
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = model.generate([prompt_text], self._get_sampling_params(**kwargs), use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    def chat_batch(self, messages_list: List[List[Dict[str, str]]], **kwargs: Any) -> List[str]:
        """
        Processes a list of conversational exchanges concurrently leveraging vLLM's continuous batching engine.
        """
        model = self._load_resources()
        tokenizer = model.get_tokenizer()
        
        formatted_prompts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]
            
        outputs = model.generate(formatted_prompts, self._get_sampling_params(**kwargs), use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]

    def release_resources(self) -> None:
        """
        Deeply purges the vLLM engine from VRAM by destroying distributed worker groups and clearing CUDA cache.
        """
        cache_key = self._get_cache_key()
        cls = self.__class__
        
        if cache_key in cls._ref_counts:
            cls._ref_counts[cache_key] -= 1
            if cls._ref_counts[cache_key] > 0:
                return  # Still in use by other instances
            else:
                del cls._ref_counts[cache_key]
                
        if cache_key in cls._model_cache:
            try:
                # Essential for vLLM: destroy parallel state before clearing cache to prevent VRAM leaks
                from vllm.distributed.parallel_state import destroy_model_group, destroy_distributed_environment
                destroy_model_group()
                destroy_distributed_environment()
            except ImportError:
                pass
                
            del cls._model_cache[cache_key]

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
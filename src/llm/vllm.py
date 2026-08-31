import gc
from typing import List, Dict, Optional, Any
from ..contracts.llm import LLM
from ..utils.io import read_text


class VllmClient(LLM):
    """
    LLM Client implementation utilizing `vLLM` for high-throughput, GPU-accelerated inference.
    Features native continuous batching and advanced VRAM management for sequential pipelines.
    """

    def __init__(self, model_name: str, backend: Dict[str, Any], prompt_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> None:
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
        self._model: Any | None = None
        self.prompt_template: str | None = None
        
        if prompt_path:
            self.prompt_template = read_text(prompt_path)

    def _load_resources(self) -> Any:
        """
        Lazy loads the vLLM Engine into memory, allocating the specified VRAM portion.
        """
        if self._model is None:
            from vllm import LLM

            b_kwargs = {k: v for k, v in self.backend.items() if k not in ["type", "gpu_memory_utilization", "max_model_len", "dtype", "quantization"]}
            self._model = LLM(
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

    def generate(self, template_kwargs: Dict[str, Any], system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """
        Generates text completion for a single prompt using chat template formatting.
        """
        messages = []
        prompt = self.prompt_template.format(**template_kwargs) if self.prompt_template else ""
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

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

    def generate_batch(self, template_kwargs_list: List[Dict[str, Any]], system_prompt: Optional[str] = None, **kwargs: Any) -> List[str]:
        """
        Processes a list of prompts concurrently leveraging vLLM's continuous batching engine.
        """
        model = self._load_resources()
        tokenizer = model.get_tokenizer()
        
        formatted_prompts = []
        for tk in template_kwargs_list:
            msgs = [{"role": "system", "content": system_prompt}] if system_prompt else []
            prompt = self.prompt_template.format(**tk) if self.prompt_template else ""
            msgs.append({"role": "user", "content": prompt})
            formatted_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            
        outputs = model.generate(formatted_prompts, self._get_sampling_params(**kwargs), use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]

    def release_resources(self) -> None:
        """
        Deeply purges the vLLM engine from VRAM by destroying distributed worker groups and clearing CUDA cache.
        """
        if self._model is not None:
            try:
                # Essential for vLLM: destroy parallel state before clearing cache to prevent VRAM leaks
                from vllm.distributed.parallel_state import destroy_model_group, destroy_distributed_environment
                destroy_model_group()
                destroy_distributed_environment()
            except ImportError:
                pass
                
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
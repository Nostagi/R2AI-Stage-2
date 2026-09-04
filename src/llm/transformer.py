import gc
from typing import Any, Dict, List, Optional
import torch

from ..contracts.llm import LLM
from ..utils.io import read_text


class TransformersClient(LLM):
    """
    Hugging Face Transformers implementation of the BaseLLMClient interface.
    
    Designed for local execution using standard PyTorch causal models.
    Supports lazy resource loading and explicit VRAM/RAM memory release.
    """
    _model_cache: Dict[str, Any] = {}
    _tokenizer_cache: Dict[str, Any] = {}
    _ref_counts: Dict[str, int] = {}

    def __init__(self, model_name: str, backend: Dict[str, Any], params: Optional[Dict[str, Any]] = None):
        """
        Initializes the Transformers client configuration.

        Args:
            model_name (str): Hugging Face model identifier or local checkpoint path.
            backend (Dict[str, Any]): Hardware & precision config 
                (e.g., {"device": "cuda", "dtype": "bfloat16"}).
            params (Optional[Dict[str, Any]]): Default inference parameters 
                (e.g., {"temperature": 0.0, "max_new_tokens": 512}).
        """
        self.model_id: str = model_name
        self.device: str = backend.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_str: str = backend.get("dtype", "auto")
        self.default_params: Dict[str, Any] = params or {}

        cache_key = self._get_cache_key()
        cls = self.__class__
        cls._ref_counts[cache_key] = cls._ref_counts.get(cache_key, 0) + 1

    def _get_cache_key(self) -> str:
        return f"{self.model_id}_{self.device}_{self.dtype_str}"

    @property
    def _model(self) -> Any:
        return self._model_cache.get(self._get_cache_key())

    @property
    def _tokenizer(self) -> Any:
        return self._tokenizer_cache.get(self._get_cache_key())

    def _parse_dtype(self, dtype_str: str) -> Any:
        """Maps string dtype representations to torch.dtype objects."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        return dtype_map.get(dtype_str.lower(), "auto")

    def _ensure_model_downloaded(self) -> None:
        hf_repo = self.backend.get("hf_repo")
        if hf_repo and self.model_id.startswith("./"):
            import os
            if not os.path.exists(self.model_id):
                print(f"Downloading {hf_repo} to {self.model_id}...")
                from huggingface_hub import snapshot_download
                from src.config import get_settings
                token = get_settings().hf_token
                snapshot_download(repo_id=hf_repo, local_dir=self.model_id, token=token)

    def _load_resources(self) -> None:
        """Lazy loads model and tokenizer into GPU/CPU memory on first request."""
        self._ensure_model_downloaded()
        cache_key = self._get_cache_key()

        if cache_key not in self._tokenizer_cache:
            from transformers import AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            # Left padding is required for batched generation in Causal LMs
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left"
            self._tokenizer_cache[cache_key] = tokenizer

        if cache_key not in self._model_cache:
            from transformers import AutoModelForCausalLM  # type: ignore

            torch_dtype = self._parse_dtype(self.dtype_str)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map=self.device if self.device != "cuda" else "auto",
                trust_remote_code=True,
            )
            self._model_cache[cache_key] = model

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Generates a response from a list of conversational messages."""
        self._load_resources()

        # Merge instance default params with per-call overrides
        gen_kwargs = {**self.default_params, **kwargs}
        temperature = float(gen_kwargs.pop("temperature", 0.0))
        max_new_tokens = int(gen_kwargs.pop("max_new_tokens", gen_kwargs.pop("max_tokens", 512)))
        top_p = gen_kwargs.pop("top_p", None)

        prompt_text = self._tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                top_p=top_p if temperature > 0.0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
                **gen_kwargs
            )

        # Slice off input prompt tokens to retrieve only the generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = output_ids[0][input_len:]

        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def chat_batch(self, messages_list: List[List[Dict[str, str]]], **kwargs: Any) -> List[str]:
        """Processes multiple conversational exchanges in a single batch."""
        if not messages_list:
            return []

        self._load_resources()

        gen_kwargs = {**self.default_params, **kwargs}
        temperature = float(gen_kwargs.pop("temperature", 0.0))
        max_new_tokens = int(gen_kwargs.pop("max_new_tokens", gen_kwargs.pop("max_tokens", 512)))
        top_p = gen_kwargs.pop("top_p", None)

        formatted_texts = [
            self._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]

        inputs = self._tokenizer(
            formatted_texts, 
            return_tensors="pt", 
            padding=True
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                top_p=top_p if temperature > 0.0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
                **gen_kwargs
            )

        # Decode responses individually based on individual prompt lengths
        input_lens = [len(ids) for ids in inputs["input_ids"]]
        results = []
        for i, out_ids in enumerate(output_ids):
            gen_tokens = out_ids[input_lens[i]:]
            decoded = self._tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            results.append(decoded)

        return results

    def release_resources(self) -> None:
        """
        Unloads model weights and tokenizer from memory, releasing VRAM/RAM 
        for subsequent sequential pipeline steps.
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
            del cls._model_cache[cache_key]

        if cache_key in cls._tokenizer_cache:
            del cls._tokenizer_cache[cache_key]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
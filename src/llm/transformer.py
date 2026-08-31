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

    def __init__(self, model_name: str, backend: Dict[str, Any], prompt_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        """
        Initializes the Transformers client configuration.

        Args:
            model_name (str): Hugging Face model identifier or local checkpoint path.
            backend (Dict[str, Any]): Hardware & precision config 
                (e.g., {"device": "cuda", "dtype": "bfloat16"}).
            prompt_path (Optional[str]): Path to prompt template file.
            params (Optional[Dict[str, Any]]): Default inference parameters 
                (e.g., {"temperature": 0.0, "max_new_tokens": 512}).
        """
        self.model_id: str = model_name
        self.device: str = backend.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_str: str = backend.get("dtype", "auto")
        self.default_params: Dict[str, Any] = params or {}
        self.prompt_template: str | None = None

        if prompt_path:
            self.prompt_template = read_text(prompt_path)

        # Lazy-loaded attributes
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

    def _parse_dtype(self, dtype_str: str) -> Any:
        """Maps string dtype representations to torch.dtype objects."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        return dtype_map.get(dtype_str.lower(), "auto")

    def _load_resources(self) -> None:
        """Lazy loads model and tokenizer into GPU/CPU memory on first request."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            # Left padding is required for batched generation in Causal LMs
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            self._tokenizer.padding_side = "left"

        if self._model is None:
            from transformers import AutoModelForCausalLM  # type: ignore

            torch_dtype = self._parse_dtype(self.dtype_str)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map=self.device if self.device != "cuda" else "auto",
                trust_remote_code=True,
            )

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

    def generate(self, template_kwargs: Dict[str, Any], system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """Generates a response for a single prompt string."""
        messages = []
        prompt = self.prompt_template.format(**template_kwargs) if self.prompt_template else ""
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, **kwargs)

    def generate_batch(self, template_kwargs_list: List[Dict[str, Any]], system_prompt: Optional[str] = None, **kwargs: Any) -> List[str]:
        """Processes multiple prompts concurrently using batched tokenization."""
        if not template_kwargs_list:
            return []

        self._load_resources()

        gen_kwargs = {**self.default_params, **kwargs}
        temperature = float(gen_kwargs.pop("temperature", 0.0))
        max_new_tokens = int(gen_kwargs.pop("max_new_tokens", gen_kwargs.pop("max_tokens", 512)))
        top_p = gen_kwargs.pop("top_p", None)

        # Format prompt list into template-formatted strings
        formatted_texts = []
        for tk in template_kwargs_list:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            prompt = self.prompt_template.format(**tk) if self.prompt_template else ""
            msgs.append({"role": "user", "content": prompt})
            
            text = self._tokenizer.apply_chat_template(
                msgs, 
                tokenize=False, 
                add_generation_prompt=True
            )
            formatted_texts.append(text)

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
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
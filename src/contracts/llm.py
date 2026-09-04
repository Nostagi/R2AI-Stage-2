from __future__ import annotations
import json
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class LLM(ABC):
    """
    Abstract Base Class defining the unified contract for LLM operations
    within the Finance QA pipeline.
    
    Clients only need to implement chat interfaces. Prompt formatting is 
    handled centrally by the LLMProvider.
    """

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Processes a conversational exchange using a standard message format.

        Args:
            messages (List[Dict[str, str]]): A sequence of messages 
                (e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]).
            **kwargs: Backend-specific sampling parameters.

        Returns:
            str: The generated response from the model.
        """
        pass

    @abstractmethod
    def chat_batch(self, messages_list: List[List[Dict[str, str]]], **kwargs: Any) -> List[str]:
        """
        Processes multiple conversational exchanges in a single batch.

        Args:
            messages_list (List[List[Dict[str, str]]]): A list of message sequences.
            **kwargs: Backend-specific sampling parameters.

        Returns:
            List[str]: A list of responses corresponding to the input prompts.
        """
        pass

    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        Generates dense embeddings for a list of input texts.
        
        Args:
            texts (List[str]): The input texts to embed.
            **kwargs: Backend-specific sampling parameters.
            
        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        raise NotImplementedError("Embeddings generation is not supported by this LLM backend.")

    @abstractmethod
    def release_resources(self) -> None:
        """
        Forcefully unloads the model from memory (VRAM/RAM) and cleans up allocated resources.
        """
        pass

class LLMProvider:
    """
    Factory, Lifecycle Manager, and Prompt Manager for LLM instances.
    Provides lazy instantiation, alias-to-client mapping, resource management,
    and centrally constructs prompts before passing them to the appropriate client.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the provider with a configuration dictionary mapping aliases to configs.
        Loads unified JSON prompts.
        """
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, LLM] = {}
        self._failed_aliases = set()
        self._lock = threading.Lock()
        self.prompts: Dict[str, Dict[str, str]] = {}
        
        prompt_path = config.get("prompt_path")
        if prompt_path:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompts = json.load(f)

        for name in prompts.keys():
            for prompt_type in prompts[name].keys():
                self.prompts[name] = {}
                self.prompts[name][prompt_type] = "\n".join(prompts[name][prompt_type])
        
        # Parse list of LLMs into an alias-indexed lookup table
        for item in config.get("llms", []):
            if "alias" in item:
                self._configs[item["alias"]] = item

    def get_llm(self, alias: str) -> LLM:
        """
        Retrieves an active LLM client by alias, instantiating it lazily if not already loaded.
        Ensures thread-safe initialization to prevent concurrent downloads/loads by multiple workers.
        """
        if alias not in self._configs:
            raise KeyError(f"LLM alias '{alias}' not found in configuration.")

        if alias in self._failed_aliases:
            raise RuntimeError(f"LLM alias '{alias}' previously failed to load (cached error). Aborting.")

        if alias not in self._instances:
            with self._lock:
                if alias in self._failed_aliases:
                    raise RuntimeError(f"LLM alias '{alias}' previously failed to load (cached error). Aborting.")
                    
                # Double-checked locking pattern
                if alias not in self._instances:
                    try:
                        self._instances[alias] = self._instantiate_client(self._configs[alias])
                    except Exception as e:
                        self._failed_aliases.add(alias)
                        raise

        return self._instances[alias]

    def generate(self, alias: str, prompt_name: str, template_kwargs: Dict[str, Any], **kwargs: Any) -> str:
        """
        Generates text by fetching a predefined prompt, formatting it, and passing to the requested LLM.
        """
        prompt_config = self.prompts.get(prompt_name, {})
        system_prompt = prompt_config.get("system_prompt", "")
        user_prompt_template = prompt_config.get("user_prompt", "")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        prompt = user_prompt_template.format(**template_kwargs) if user_prompt_template else ""
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(alias, messages, **kwargs)

    def generate_batch(self, alias: str, prompt_name: str, template_kwargs_list: List[Dict[str, Any]], **kwargs: Any) -> List[str]:
        """
        Generates a batch of texts using a predefined prompt and multiple kwargs contexts.
        """
        prompt_config = self.prompts.get(prompt_name, {})
        system_prompt = prompt_config.get("system_prompt", "")
        user_prompt_template = prompt_config.get("user_prompt", "")

        messages_list = []
        for tk in template_kwargs_list:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            prompt = user_prompt_template.format(**tk) if user_prompt_template else ""
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)

        return self.chat_batch(alias, messages_list, **kwargs)

    def chat(self, alias: str, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Passes a raw chat completion request to the chosen LLM.
        Supports 'time_limit' (in seconds) to abort waiting if generation takes too long.
        """
        client = self.get_llm(alias)
        time_limit = kwargs.pop("time_limit", None)
        
        if time_limit is not None:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(client.chat, messages, **kwargs)
                try:
                    return future.result(timeout=time_limit)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"LLM generation exceeded the time limit of {time_limit} seconds.")
                    
        return client.chat(messages, **kwargs)

    def chat_batch(self, alias: str, messages_list: List[List[Dict[str, str]]], **kwargs: Any) -> List[str]:
        """
        Passes a batch raw chat completion request to the chosen LLM.
        Supports 'time_limit' (in seconds) to abort waiting if batch generation takes too long.
        """
        client = self.get_llm(alias)
        time_limit = kwargs.pop("time_limit", None)
        
        if time_limit is not None:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(client.chat_batch, messages_list, **kwargs)
                try:
                    return future.result(timeout=time_limit)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"LLM batch generation exceeded the time limit of {time_limit} seconds.")

        return client.chat_batch(messages_list, **kwargs)

    def embed(self, alias: str, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using the chosen LLM backend.
        Supports 'time_limit' (in seconds) to abort waiting if generation takes too long.
        """
        client = self.get_llm(alias)
        time_limit = kwargs.pop("time_limit", None)
        
        if time_limit is not None:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(client.embed, texts, **kwargs)
                try:
                    return future.result(timeout=time_limit)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"LLM embedding exceeded the time limit of {time_limit} seconds.")

        return client.embed(texts, **kwargs)

    def release_llm(self, alias: str) -> None:
        """
        Forces the specific LLM instance associated with the alias to release VRAM/RAM.
        """
        if alias in self._instances:
            self._instances[alias].release_resources()
            del self._instances[alias]

    def release_all(self) -> None:
        """
        Releases VRAM/RAM resources for all currently active LLM instances.
        """
        for alias in list(self._instances.keys()):
            self.release_llm(alias)

    def _instantiate_client(self, cfg: Dict[str, Any]) -> LLM:
        """
        Internal factory method mapping backend configuration types to LLM client objects.
        """
        backend_cfg = cfg.get("backend", {})
        params = cfg.get("params", {})
        backend_type = backend_cfg.get("type", "").lower()
        model_name = cfg.get("model", "")

        match backend_type:
            case "transformers":
                from ..llm.transformer import TransformersClient

                return TransformersClient(
                    model_name=model_name,
                    backend=backend_cfg,
                    params=params
                )

            case "llama_cpp":
                from ..llm.llamacpp import LlamaCppClient

                return LlamaCppClient(
                    model_name=model_name,
                    backend=backend_cfg,
                    params=params
                )

            case "vllm":
                from ..llm.vllm import VllmClient

                return VllmClient(
                    model_name=model_name,
                    backend=backend_cfg,
                    params=params
                )

            case "ollama":
                from ..llm.ollama import OllamaClient

                return OllamaClient(
                    model_name=model_name,
                    backend=backend_cfg,
                    params=params
                )

            case "openai" | "openai_compatible" | "api":
                from ..llm.openAPI import OpenAICompatibleClient

                return OpenAICompatibleClient(
                    model_name=model_name,
                    backend=backend_cfg,
                    params=params
                )

            case _:
                raise ValueError(f"Unsupported backend type '{backend_type}' for alias '{cfg.get('alias')}'")
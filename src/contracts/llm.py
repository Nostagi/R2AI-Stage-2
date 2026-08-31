from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from __future__ import annotations
import yaml


class LLM(ABC):
    """
    Abstract Base Class defining the unified contract for LLM operations
    within the Finance QA pipeline.
    
    This interface ensures seamless interchangeability across various local 
    and remote backends (e.g., vLLM, Llama.cpp, Ollama, OpenAI-compatible APIs).
    It also provides resource management capabilities for sequential pipelines 
    running on constrained GPU environments.
    """

    @abstractmethod
    def generate(self, template_kwargs: Dict[str, Any], system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """
        Generates a text completion for a single prompt.
        Ideal for straightforward extraction or broad candidate generation (Recall).

        Args:
            template_kwargs (Dict[str, Any]): The dictionary containing values to format the prompt template.
            system_prompt (Optional[str]): Optional system context to guide the model's behavior.
            **kwargs: Backend-specific sampling parameters (e.g., temperature, top_p, max_tokens).

        Returns:
            str: The generated text output.
        """
        pass

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
    def generate_batch(self, template_kwargs_list: List[Dict[str, Any]], system_prompt: Optional[str] = None, **kwargs: Any) -> List[str]:
        """
        Processes multiple prompts in a single batch.

        Args:
            template_kwargs_list (List[Dict[str, Any]]): A list of dictionaries to format the prompt template.
            system_prompt (Optional[str]): A shared system prompt applied to all inputs.
            **kwargs: Backend-specific sampling parameters.

        Returns:
            List[str]: A list of responses corresponding to the input prompts.
        """
        pass

    @abstractmethod
    def release_resources(self) -> None:
        """
        Forcefully unloads the model from memory (VRAM/RAM) and cleans up allocated resources.
        """
        pass

class LLMProvider:
    """
    Factory and Lifecycle Manager for LLM instances defined in system configuration.
    Provides lazy instantiation, alias-to-client mapping, and resource management.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the provider with a configuration dictionary mapping aliases to configs.
        """
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, LLM] = {}
        
        # Parse list of LLMs into an alias-indexed lookup table
        for item in config.get("llms", []):
            if "alias" in item:
                self._configs[item["alias"]] = item

    def get_llm(self, alias: str) -> LLM:
        """
        Retrieves an active LLM client by alias, instantiating it lazily if not already loaded.
        """
        if alias not in self._configs:
            raise KeyError(f"LLM alias '{alias}' not found in configuration.")

        if alias not in self._instances:
            self._instances[alias] = self._instantiate_client(self._configs[alias])

        return self._instances[alias]

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
        prompt_path = cfg.get("prompt_path")

        match backend_type:
            case "transformers":
                from ..llm.transformer import TransformersClient

                return TransformersClient(
                    model_name=model_name,
                    backend=backend_cfg,
                    prompt_path=prompt_path,
                    params=params
                )

            case "llama_cpp":
                from ..llm.llamacpp import LlamaCppClient

                return LlamaCppClient(
                    model_name=model_name,
                    backend=backend_cfg,
                    prompt_path=prompt_path,
                    params=params
                )

            case "vllm":
                from ..llm.vllm import VllmClient

                return VllmClient(
                    model_name=model_name,
                    backend=backend_cfg,
                    prompt_path=prompt_path,
                    params=params
                )

            case "ollama":
                from ..llm.ollama import OllamaClient

                return OllamaClient(
                    model_name=model_name,
                    backend=backend_cfg,
                    prompt_path=prompt_path,
                    params=params
                )

            case "openai" | "openai_compatible" | "api":
                from ..llm.openAPI import OpenAICompatibleClient

                return OpenAICompatibleClient(
                    model_name=model_name,
                    backend=backend_cfg,
                    prompt_path=prompt_path,
                    params=params
                )

            case _:
                raise ValueError(f"Unsupported backend type '{backend_type}' for alias '{cfg.get('alias')}'")
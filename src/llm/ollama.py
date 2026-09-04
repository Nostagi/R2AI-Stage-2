from typing import List, Dict, Optional, Any
from ..contracts.llm import LLM
from ..utils.io import read_text


class OllamaClient(LLM):
    """
    LLM Client implementation utilizing Ollama for local low-latency model execution.
    Supports explicit VRAM offloading via Ollama's keep_alive parameter.
    """
    _ref_counts: Dict[str, int] = {}

    def __init__(self, model_name: str, backend: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes configuration parameters and prepares the lazy-loaded Ollama client.
        """
        self.model_name = model_name
        self.host = backend.get("host", "http://localhost:11434")
        self.backend = backend
        self.default_params = params or {}
        self._client: Any | None = None

        cache_key = self.model_name
        cls = self.__class__
        cls._ref_counts[cache_key] = cls._ref_counts.get(cache_key, 0) + 1

    def _get_client(self) -> Any:
        """
        Lazy-initializes the Ollama API Client instance.
        """
        if self._client is None:
            from ollama import Client

            b_kwargs = {k: v for k, v in self.backend.items() if k not in ["type", "host"]}
            self._client = Client(host=self.host, **b_kwargs)
            
            try:
                self._client.show(self.model_name)
            except Exception as e:
                if "not found" in str(e).lower():
                    print(f"Downloading model {self.model_name} from Ollama...")
                    self._client.pull(self.model_name)
        return self._client

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Executes a chat completion via the Ollama REST API.
        """
        client = self._get_client()
        gen_kwargs = {**self.default_params, **kwargs}
        temp = gen_kwargs.pop("temperature", 0.0)
        num_predict = gen_kwargs.pop("num_predict", gen_kwargs.pop("max_tokens", gen_kwargs.pop("max_new_tokens", 1024)))

        options = {
            "temperature": temp,
            "num_predict": num_predict,
            **gen_kwargs
        }

        response = client.chat(
            model=self.model_name,
            messages=messages,
            options=options
        )
        return response["message"]["content"] or ""

    def chat_batch(self, messages_list: List[List[Dict[str, str]]], **kwargs: Any) -> List[str]:
        """
        Sequentially processes a batch of conversational exchanges through the Ollama client.
        """
        return [self.chat(msgs, **kwargs) for msgs in messages_list]

    def release_resources(self) -> None:
        """
        Forces Ollama to immediately unload the model from VRAM by setting keep_alive to 0.
        """
        cache_key = self.model_name
        cls = self.__class__
        
        if cache_key in cls._ref_counts:
            cls._ref_counts[cache_key] -= 1
            if cls._ref_counts[cache_key] > 0:
                self._client = None
                return
            else:
                del cls._ref_counts[cache_key]

        if self._client is not None:
            try:
                # Setting keep_alive=0 prompts Ollama to purge the model from memory instantly
                self._client.generate(model=self.model_name, keep_alive=0)
            except Exception:
                pass
            self._client = None
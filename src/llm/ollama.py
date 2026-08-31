from typing import List, Dict, Optional, Any
from ..contracts.llm import LLM
from ..utils.io import read_text


class OllamaClient(LLM):
    """
    LLM Client implementation utilizing Ollama for local low-latency model execution.
    Supports explicit VRAM offloading via Ollama's keep_alive parameter.
    """

    def __init__(self, model_name: str, backend: Dict[str, Any], prompt_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes configuration parameters and prepares the lazy-loaded Ollama client.
        """
        self.model_name = model_name
        self.host = backend.get("host", "http://localhost:11434")
        self.backend = backend
        self.default_params = params or {}
        self._client: Any | None = None
        self.prompt_template: str | None = None

        # Load nội dung prompt từ file text nếu đường dẫn được khai báo
        if prompt_path:
            self.prompt_template = read_text(prompt_path)

    def _get_client(self) -> Any:
        """
        Lazy-initializes the Ollama API Client instance.
        """
        if self._client is None:
            from ollama import Client

            b_kwargs = {k: v for k, v in self.backend.items() if k not in ["type", "host"]}
            self._client = Client(host=self.host, **b_kwargs)
        return self._client

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

    def generate_batch(self, template_kwargs_list: List[Dict[str, Any]], system_prompt: Optional[str] = None, **kwargs: Any) -> List[str]:
        """
        Sequentially processes a batch of prompts through the Ollama client.
        """
        return [self.generate(tk, system_prompt=system_prompt, **kwargs) for tk in template_kwargs_list]

    def release_resources(self) -> None:
        """
        Forces Ollama to immediately unload the model from VRAM by setting keep_alive to 0.
        """
        if self._client is not None:
            try:
                # Setting keep_alive=0 prompts Ollama to purge the model from memory instantly
                self._client.generate(model=self.model_name, keep_alive=0)
            except Exception:
                pass
            self._client = None
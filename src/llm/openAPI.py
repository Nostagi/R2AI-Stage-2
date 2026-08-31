from typing import List, Dict, Optional, Any
from ..contracts.llm import LLM
from ..utils.io import read_text


class OpenAICompatibleClient(LLM):
    """
    LLM Client implementation for OpenAI-compatible enterprise REST APIs or hosted servers (vLLM / TGI).
    Handles remote API requests where memory management is offloaded to the server infrastructure.
    """

    def __init__(self, model_name: str, backend: Dict[str, Any], prompt_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes API endpoint parameters and client authentication configs.
        """
        self.model_name = model_name
        self.base_url = backend.get("base_url", "http://localhost:8000/v1")
        self.api_key = backend.get("api_key", "EMPTY")
        self.backend = backend
        self.default_params = params or {}
        self._client: Any | None = None
        self.prompt_template: str | None = None
        
        if prompt_path:
            self.prompt_template = read_text(prompt_path)

    def _get_client(self) -> Any:
        """
        Lazy-initializes the OpenAI client instance.
        """
        if self._client is None:
            from openai import OpenAI

            b_kwargs = {k: v for k, v in self.backend.items() if k not in ["type", "base_url", "api_key"]}
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                **b_kwargs
            )
        return self._client

    def generate(self, template_kwargs: Dict[str, Any], system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """
        Generates text completion for a single prompt by routing to the chat endpoint.
        """
        messages = []
        prompt = self.prompt_template.format(**template_kwargs) if self.prompt_template else ""
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Executes chat completion via the OpenAI-compatible REST endpoint.
        """
        client = self._get_client()
        gen_kwargs = {**self.default_params, **kwargs}
        temp = gen_kwargs.pop("temperature", 0.0)
        max_tok = gen_kwargs.pop("max_tokens", gen_kwargs.pop("max_new_tokens", 1024))
        top_p_val = gen_kwargs.pop("top_p", 1.0)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
            top_p=top_p_val,
            **gen_kwargs
        )
        return (response.choices[0].message.content or "").strip()

    def generate_batch(self, template_kwargs_list: List[Dict[str, Any]], system_prompt: Optional[str] = None, **kwargs: Any) -> List[str]:
        """
        Sequentially executes batch requests. Can be extended with asyncio/threading for concurrency.
        """
        return [self.generate(tk, system_prompt=system_prompt, **kwargs) for tk in template_kwargs_list]

    def release_resources(self) -> None:
        """
        Closes the HTTP client session. Server-side VRAM management is handled remotely.
        """
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
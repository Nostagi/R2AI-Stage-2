from typing import List, Dict, Optional, Any
from ..contracts.llm import LLM
from ..utils.io import read_text


class OpenAICompatibleClient(LLM):
    """
    LLM Client implementation for OpenAI-compatible enterprise REST APIs or hosted servers (vLLM / TGI).
    Handles remote API requests where memory management is offloaded to the server infrastructure.
    """

    def __init__(self, model_name: str, backend: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes API endpoint parameters and client authentication configs.
        """
        self.model_name = model_name
        self.base_url = backend.get("base_url", "http://localhost:8000/v1")
        self.api_key = backend.get("api_key", "EMPTY")
        self.backend = backend
        self.default_params = params or {}
        self._client: Any | None = None

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

    def chat_batch(self, messages_list: List[List[Dict[str, str]]], **kwargs: Any) -> List[str]:
        """
        Sequentially executes batch requests. Can be extended with asyncio/threading for concurrency.
        """
        return [self.chat(msgs, **kwargs) for msgs in messages_list]

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
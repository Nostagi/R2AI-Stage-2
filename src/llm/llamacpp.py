import gc
from typing import List, Dict, Optional, Any
from ..contracts.llm import LLM
from ..utils.io import read_text


class LlamaCppClient(LLM):
    """
    LLM Client implementation utilizing `llama-cpp-python` for local GGUF model execution.
    Designed for memory efficiency and configurable GPU layer offloading.
    """
    _model_cache: Dict[str, Any] = {}
    _ref_counts: Dict[str, int] = {}

    def __init__(self, model_name: str, backend: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the LlamaCppClient configuration parameters.
        """
        raw_file = backend.get("model_file", model_name)
        
        import os
        from src.config import get_settings
        models_dir = get_settings().paths.models
        
        # Nếu model_file chỉ là tên file hoặc relative path, tự động gộp với settings.paths.models
        self.model_file = raw_file
            
        self.n_ctx = backend.get("n_ctx", 8192)
        self.n_gpu_layers = backend.get("n_gpu_layers", -1)
        self.backend = backend
        self.default_params = params or {}

        # Pre-download check inside the centralized LLMProvider thread-lock
        self._ensure_model_downloaded()

        cache_key = self._get_cache_key()
        cls = self.__class__
        cls._ref_counts[cache_key] = cls._ref_counts.get(cache_key, 0) + 1

    def _get_cache_key(self) -> str:
        return f"{self.model_file}_{self.n_ctx}_{self.n_gpu_layers}"

    @property
    def _model(self) -> Any:
        return self._model_cache.get(self._get_cache_key())

    def _ensure_model_downloaded(self) -> None:
        hf_repo = self.backend.get("hf_repo")
        hf_file = self.backend.get("hf_file")
        if hf_repo and hf_file:
            import os
            if not os.path.exists(self.model_file):
                print(f"Downloading {hf_file} from {hf_repo}...")
                from huggingface_hub import hf_hub_download
                from src.config import get_settings
                token = get_settings().hf_token
                local_dir = os.path.dirname(self.model_file)
                os.makedirs(local_dir, exist_ok=True)
                try:
                    hf_hub_download(repo_id=hf_repo, filename=hf_file, local_dir=local_dir, token=token)
                except Exception as e:
                    import shutil
                    # Cleanup thư mục rỗng nếu tải hỏng
                    if os.path.exists(local_dir) and not os.listdir(local_dir):
                        shutil.rmtree(local_dir, ignore_errors=True)
                    raise RuntimeError(f"Không thể tải model '{hf_file}' từ repo '{hf_repo}'. Có thể file không tồn tại hoặc server từ chối. Lỗi chi tiết: {e}")

    def _load_resources(self) -> Any:
        """
        Lazy loads the Llama model instance into memory upon first invocation.
        """
        cache_key = self._get_cache_key()

        if cache_key not in self._model_cache:
            from llama_cpp import Llama

            b_kwargs = {k: v for k, v in self.backend.items() if k not in ["type", "model_file", "n_ctx", "n_gpu_layers"]}
            # Force embedding=True to support bi-encoder / dense embedding
            b_kwargs["embedding"] = True
            
            self._model_cache[cache_key] = Llama(
                model_path=self.model_file,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                **b_kwargs
            )
        return self._model

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Executes a multi-turn chat sequence using native Llama.cpp chat formatting.
        """
        model = self._load_resources()
        gen_kwargs = {**self.default_params, **kwargs}
        temp = gen_kwargs.pop("temperature", 0.0)
        max_tok = gen_kwargs.pop("max_tokens", gen_kwargs.pop("max_new_tokens", 512))

        response = model.create_chat_completion(
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
            **gen_kwargs
        )
        return response["choices"][0]["message"]["content"] or ""

    def chat_batch(self, messages_list: List[List[Dict[str, str]]], **kwargs: Any) -> List[str]:
        """
        Sequentially executes chat generation for a list of conversations.
        """
        return [self.chat(msgs, **kwargs) for msgs in messages_list]

    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        Generates dense embeddings for a list of texts using llama.cpp.
        """
        model = self._load_resources()
        embeddings = []
        for text in texts:
            # llama.cpp create_embedding trả về dict chứa data list
            res = model.create_embedding(text, **kwargs)
            embed_vec = res["data"][0]["embedding"]
            embeddings.append(embed_vec)
        return embeddings

    def release_resources(self) -> None:
        """
        Unloads the GGUF model from VRAM/RAM and invokes garbage collection.
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

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
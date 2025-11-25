import os
import json
import asyncio
from uuid import uuid4
from typing_extensions import AsyncGenerator
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
# from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import RequestOutputKind
from vllm.outputs import CompletionOutput
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
import torch.distributed as dist


os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/runpod-volume/torch_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/runpod-volume/triton_cache")
os.environ.setdefault("VLLM_CACHE_DIR", "/runpod-volume/vllm_cache")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_EAGER", "0")
os.environ.setdefault("GPU_MEM_UTILIZATION", "0.9")
os.environ.setdefault("MAX_TOKENS", "4096")
os.environ.setdefault("MAX_CONCURRENT_REQUESTS", "128")
os.environ.setdefault("MAX_CONCURRENT_TOKENS", "4096")
# os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")


class vLLMService:
    _async_llm_engine: AsyncLLM
    _use_eager: bool = os.getenv("VLLM_EAGER", "0") == "1"
    _compilation_config: CompilationConfig
    _args: AsyncEngineArgs
    _engine_ready: bool = False

    @classmethod
    async def init_resource(cls) -> AsyncLLM:
        if not cls._use_eager:
            cls._compilation_config = CompilationConfig(
                level=3,
                cache_dir="/runpod-volume/vllm_cache",
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
                cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            )
            cls._args = AsyncEngineArgs(
                model="./models/language_model/GRPO-Vi-Qwen2-7B-RAG-W4A16",
                tokenizer="./models/language_model/GRPO-Vi-Qwen2-7B-RAG-W4A16",
                quantization='compressed-tensors',
                dtype="bfloat16",
                max_model_len=int(
                    os.getenv("MAX_TOKENS", "4096")
                ),
                gpu_memory_utilization=float(
                    os.getenv("GPU_MEM_UTILIZATION", "0.9")
                ),
                max_num_seqs=int(
                    os.getenv("MAX_CONCURRENT_REQUESTS", "128")
                ),
                max_num_batched_tokens=int(
                    os.getenv("MAX_CONCURRENT_TOKENS", "4096")
                ),
                enable_chunked_prefill=True,
                enable_prefix_caching=True,
                swap_space=4,
                enforce_eager=cls._use_eager,
                compilation_config=cls._compilation_config
            )
        else:
            cls._args = AsyncEngineArgs(
                model="./models/language_model/GRPO-Vi-Qwen2-7B-RAG-W4A16",
                tokenizer="./models/language_model/GRPO-Vi-Qwen2-7B-RAG-W4A16",
                quantization='compressed-tensors',
                dtype="bfloat16",
                max_model_len=int(
                    os.getenv("MAX_TOKENS", "4096")
                ),
                gpu_memory_utilization=float(
                    os.getenv("GPU_MEM_UTILIZATION", "0.9")
                ),
                max_num_seqs=int(
                    os.getenv("MAX_CONCURRENT_REQUESTS", "128")
                ),
                max_num_batched_tokens=int(
                    os.getenv("MAX_CONCURRENT_TOKENS", "4096")
                ),
                enable_chunked_prefill=True,
                enable_prefix_caching=True,
                swap_space=4,
                enforce_eager=cls._use_eager
            )
        cls._async_llm_engine = AsyncLLM.from_engine_args(
            engine_args=cls._args
        )
        async for _ in cls._async_llm_engine.generate(
            prompt="""\
<|im_start|>system
Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. \
Hãy luôn trả lời một cách hữu ích nhất có thể.<|im_end|>
<|im_start|>user
Chú ý các yêu cầu sau:
- Hãy trả lời bằng tiếng Việt.

### Cấu hỏi :
Xin chào!

### Trả lời :<|im_end|>
<|im_start|>assistant
""",
            sampling_params=SamplingParams(
                max_tokens=4,
                temperature=0.1
            ),
            request_id=str(uuid4())
        ):
            continue
        cls._engine_ready = True
        return cls._async_llm_engine

    @classmethod
    async def generate_answer(
        cls,
        prompt: str
    ) -> AsyncGenerator[str, None]:
        num_tokens: int = 0
        tokens: list[str] = ["",]
        try:
            assert cls._engine_ready, "LLM Engine is not ready."
            answer_generator = cls._async_llm_engine.generate(
                prompt=prompt,
                sampling_params=SamplingParams(
                    # n=1,
                    # best_of=1,
                    max_tokens=2048,
                    temperature=0.5,
                    top_p=0.85,
                    top_k=25,
                    repetition_penalty=1.1,
                    output_kind=RequestOutputKind.DELTA
                    # stop=["<|im_end|>"]
                ),
                request_id=str(uuid4())
            )
            async for request_output in answer_generator:
                completion_output: CompletionOutput = request_output.outputs[0]
                generated_text: str = completion_output.text
                if generated_text:
                    tokens.append(generated_text)
                    num_tokens += 1
                    text = tokens[(-1 * (num_tokens > 1))]
                    answer_part = {
                        "type": "answer_part",
                        "data": text
                    }
                    # print(f"data: {json.dumps(answer_part)}\n\n")
                    yield f"data: {json.dumps(answer_part)}\n\n"
                if request_output.finished:
                    stream_end = {
                        "type": "done"
                    }
                    yield f"data: {json.dumps(stream_end)}"
                    return
        except Exception as e:
            answer_part = {
                "type": "answer_part",
                "data": (
                    "[ LLM Engine died. Reinitializing ... "
                    "Try again in a few moments. ]"
                )
            }
            yield f"data: {json.dumps(answer_part)}\n\n"
            stream_end = {
                "type": "done"
            }
            yield f"data: {json.dumps(stream_end)}"
            if cls._engine_ready:
                print(f"Error during generation: {e}")
                print("Attempting to reinitialize the LLM engine...")
                cls._engine_ready = False
                cls._async_llm_engine.shutdown()
                # cls._async_llm_engine = None
                try:
                    if dist.is_available() and dist.is_initialized():
                        dist.destroy_process_group()
                except Exception as e:
                    # don't let shutdown fail because of cleanup
                    print(f"Error during torch cleanup: {e}")
                    pass
                cls._async_llm_engine = await asyncio.to_thread(
                    AsyncLLM.from_engine_args,
                    engine_args=cls._args
                )
                cls._engine_ready = True
            return

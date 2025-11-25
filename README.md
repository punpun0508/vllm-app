# vLLM serverless app

- Works with **Runpod** serverless endpoint.

**Do before building image:**

- Edit `config.json` to change:
  - `model_path`: put model's file inside `models/language_model/<model_name>`.
  - `cuda_graph_mode`: can be `full_and_piecewise`, `full_decode_only`, `full` or `piecewise`, `full_and_piecewise` should give best performance but uses lots of memory, set `cuda_graph_capture_sizes` to keep memory ultilization under control.
  - `cuda_graph_capture_sizes`: list of cuda graph sizes to be compiled on first start up if not in `eager mode`, vLLM uses graphs to speed up token processing, graphs are used dynamically wherever they fit, a graph is fixed size, when used speed up processing for the exact amount of tokens as it's size, graphs consumes GPU memory.

**With **Runpod** serverless it is possible to set environmental variables to modify vLLM Engine parameters:**

- `VLLM_EAGER`: **1** or **0**, decides whether vLLM Engine starts in eager mode for faster startup, but trades runtime processing speed (less tokens/s).
- `GPU_MEM_ULTIIZATION`: **0.0** to **1.0**, decides amount (percentage) of GPU VRAM vLLM Engine is allowed to use, more memory -> more tokens/s.
- `MAX_TOKENS`: default=**2048**, decides the context length, number of tokens the LLM "remembers" including prompts and answers.
- `MAX_CONCURRENT_REQUESTS`: default=**256**, decides the amount of requests that vLLM will attemp to process at the same time.
- `MAX_CONCURRENT_TOKENS`: default=**1024**, decides the number of tokens across all requests vLLM will batch and process at the same time.

**Local System Requirement:**

- Must have an **NVIDIA** GPU with sufficient VRAM for your model.
- NVIDIA GPU Driver with **CUDA 12.X** support.
- **nvidia-container-toolkit** to pass GPUs to containers.

**Build and run locally:**

```bash
git clone https://github.com/punpun0508/vllm-app.git
docker build -t vllm-app:latest .
# Run on all GPUs (Untested)
docker run --gpus all -p 8080:8080 -p 8081:8081 --name vllm-app vllm-app:latest
# Run on 1 GPU
docker run --gpus device=0 -p 8080:8080 -p 8081:8081 --name vllm-app vllm-app:latest
```

# vLLM Application

- Works with **Runpod** serverless endpoint.

**Do before building image:**

- Edit `init_prompt.txt` to change the initialization / warmup prompt appropriately for the chosen LLM.

- Edit `config.json` to change:
  - `model_path`: put model's file inside `models/<model_name>`.
  - `cuda_graph_mode`: can be `full_and_piecewise`, `full_decode_only`, `full` or `piecewise`, `full_and_piecewise` should give best performance but uses lots of memory, set `cuda_graph_capture_sizes` to keep memory ultilization under control, `piecewise` should use the least amount of memory.
  - `cuda_graph_capture_sizes`: list of cuda graph sizes to be compiled on first start up if not in `eager mode`, vLLM uses graphs to speed up processing, graphs are used dynamically wherever they fit, a graph is fixed size, when used speed up processing for the exact amount of tokens as it's size, graphs consumes GPU memory.
  - `keep_first_token`: whether or not to keep the first generated token.
  - `max_tokens`: the maximum allowed length of an answer.
  - `temperature`: **0.0** to **1.0**, the "creativity" slider, higher makes the LLM generate more random responses.
  - `top_p`: **0.0** to **1.0**, the maximum total probability of top tokens to consider final phase of generation, use like a budget, higher means more tokens can be squeezed in to the `"top tokens to be randomly picked"` list.
  - `top_k`: **0.0** to **1.0**, the maximum number of tokens in the `"top tokens to be randomly picked"` list during the final phase of initialization, used as a hard ceiling, example: `top_p` can be set to **1.0** to allow lots of tokens, but `top_k` will limit the amount of tokens that can be chosen.
  - `repetition_penalty`: a **floating point value** set to **> 1.0** to avoid erroneous repeated tokens generation.

**With Runpod serverless it is possible to set environmental variables to modify vLLM Engine parameters:**

- `VLLM_EAGER`: **1** or **0**, decides whether vLLM Engine starts in eager mode for faster startup, but trades runtime processing speed (less tokens/s).
- `GPU_MEM_ULTIIZATION`: **0.0** to **1.0**, decides amount (percentage) of GPU VRAM vLLM Engine is allowed to use, more memory -> more tokens/s.
- `MAX_TOKENS`: default=**2048**, decides the context length, number of tokens the LLM "remembers" including prompts and answers.
- `MAX_CONCURRENT_REQUESTS`: default=**256**, decides the amount of requests that vLLM will attemp to process at the same time.
- `MAX_CONCURRENT_TOKENS`: default=**1024**, decides the number of tokens across all requests vLLM will batch and process at the same time.

**Local System Requirement:**

- Must have an **NVIDIA** GPU with sufficient VRAM for your model.
- NVIDIA GPU Driver with at least **CUDA 12.X** support (tested with CUDA 12.8).
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

**Return format:**

- Stream of server-sent events (SSEs) as strings:

    **Token events:**
    `"data: {"type": "answer_part", "data": "token_goes_here"}"`

    **When the token stream is over (finished answering):**
    `"data: {"type": "done"}"`

- Remove the 6 characters prefix, then read data from tail as json:

```python
data = json.loads(line[6:])
# Process data
```

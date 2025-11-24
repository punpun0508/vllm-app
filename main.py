import os
from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
from api.v1 import routes as v1_routes
from contextlib import asynccontextmanager
from services.vllm_service import vLLMService
import torch.distributed as dist


os.environ.setdefault("PORT", "8080")
os.environ.setdefault("PORT_HEALTH", "8081")
ready: bool = False


###############################################################################
#
#          Manage resources with lifespan for safe shutdown and cleanup
#
###############################################################################
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vllm_engine = await vLLMService.init_resource()
    global ready
    ready = True
    try:
        yield
    finally:
        ready = False
        app.state.vllm_engine.shutdown()
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            # don't let shutdown fail because of cleanup
            print(f"Error during torch cleanup: {e}")
            pass
        # app.state.vllm_engine.shutdown()


###############################################################################
#
#                      Initialize FastAPI application
#
###############################################################################
app = FastAPI(title="AI Chatbot API", lifespan=lifespan)

# CORS setup for Streamlit frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8501"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Include versioned routers
app.include_router(v1_routes.router, prefix="/api/v1")


@app.get("/ping")
def health_check():
    return {"": 204} if not ready else {"status": "healthy"}

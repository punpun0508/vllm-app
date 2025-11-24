from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from api.v1.models import ChatRequest
from services.vllm_service import vLLMService


router = APIRouter()


###############################################################################
#
#                               API Endpoints
#
###############################################################################
@router.post("/ask")
async def ask(
    request: ChatRequest
) -> StreamingResponse:
    question = request.question.strip()
    return StreamingResponse(
        vLLMService.generate_answer(
            prompt=question
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # "Access-Control-Allow-Origin": "*",  # For CORS if needed
        }
    )

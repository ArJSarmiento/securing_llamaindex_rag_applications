from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from llama_index_rag.main import prompt

app = FastAPI()


@app.post(
    "/",
    response_class=StreamingResponse,
    response_model=str,
    summary="Query a text prompt",
    description="Query a text prompt to retrieve relevant information.",
    response_description="The retrieved information.",
)
async def main(
    input: str = Query(..., title="Input text", description="The input text to query"),
):
    return StreamingResponse(prompt(input))

import numpy as np
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc

# Initialize LightRAG with Ollama model
WORKING_DIR = "./gov_china"

WORKING_DIR = "./gov_china"
Document = "./China_The_Personal_Information_Protection_Law.md"

EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
print("api_key")
print(os.getenv("gemini_api_key"))

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gemini-1.5-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("gemini_api_key"),
        base_url="https://generativelanguage.googleapis.com/v1beta/",
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="text-embedding-004",
        api_key=os.getenv("gemini_api_key"),
        base_url="https://generativelanguage.googleapis.com/v1beta/"
    )

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=embedding_func
    )
)


with open(Document) as f:
    rag.insert(f.read())
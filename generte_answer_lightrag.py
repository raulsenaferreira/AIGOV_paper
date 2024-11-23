import numpy as np
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc


WORKING_DIR = "./gov_china"

EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
question = """these are the characteristics of my sw
{
"use_case": "VRU Detection",
"description": "Detect vulnerable road users (pedestrians, cyclists, etc.) in video from vehicle cameras.",
"data": "Images and videos of road scenes with labeled VRUs.",
"location": "Cloud storage or local servers.",
"application": "Advanced Driver-Assistance Systems (ADAS)."
}
How is it complaiant to data requirement legislation in china?"""


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


with open("./China_The_Personal_Information_Protection_Law.md") as f:
    rag.insert(f.read())


def write_to_md(file, section_title, content):
    """
    Writes a section to the Markdown file with the specified title and content.

    Args:
        file (file object): The open file object to write to.
        section_title (str): The title of the section.
        content (str): The content to include in the section.
    """
    file.write(f"## {section_title}\n")
    file.write(f"{content}\n\n")

with open("q2_ex1.md", "w") as md_file:
    # Write the title of the discussion
    md_file.write(f"# question: {question}\n\n")

    # Perform naive search and write the result
    naive_result = rag.query(question, param=QueryParam(mode="naive"))
    write_to_md(md_file, "Naive Search", naive_result)

    # Perform local search and write the result
    local_result = rag.query(question, param=QueryParam(mode="local"))
    write_to_md(md_file, "Local Search", local_result)

    # Perform global search and write the result
    global_result = rag.query(question, param=QueryParam(mode="global"))
    write_to_md(md_file, "Global Search", global_result)

    # Perform hybrid search and write the result
    hybrid_result = rag.query(question, param=QueryParam(mode="hybrid"))
    write_to_md(md_file, "Hybrid Search", hybrid_result)
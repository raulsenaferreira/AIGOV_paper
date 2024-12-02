

import numpy as np
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
from langchain_core.tools import tool
from langchain.agents import Tool 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentType, initialize_agent
from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor

from langchain.agents import AgentExecutor, create_react_agent

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
WORKING_DIR = "./gov_br"

EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
sys_prompt = """You are an AI compliance expert."""
question = """What are the key elements, provisions, and requirements outlined in this legislation related to data, including compliance obligations, rights and responsibilities of stakeholders, and any sector-specific considerations? 
"""

prompt_react = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:

------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```

Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No

Final Answer: [your response here]

```

Begin!

Previous conversation history:

{chat_history}

New input: {input}

{agent_scratchpad}"""

answer_path = "q6_ex3.md"


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
folder_country = {"brazil": "./gov_br", "china":"./gov_china"}
import json
@tool
def search_legislation(input_data):
    """
    ask a question about a legislation document of a single country
    Args:
        input_data: A dictionary with the keys:
            - question (str): The question to ask the document.
            - country (str): The country of the legislation to search for (e.g., France, Germany, Argentina).
    """
    if isinstance(input_data, str):  # Check if input is a JSON string
        input_data = json.loads(input_data)
    question = input_data.get("question")
    country = input_data.get("country")
    
    WORKING_DIR = folder_country[country.lower()]
    print()
    print(f"country: {country}, folder: {folder_country[country.lower()]}")
    print(f"question : {question}")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=embedding_func
        )
    )
    # Validate inputs
    if not question or not country:
        raise ValueError("Both 'question' and 'country' are required.")
    result = rag.query(question, param=QueryParam(mode="hybrid")) 
    print(result)
    return result

tools = [
    Tool(
        name="search",
        func=search_legislation,
        description="useful when searching for information in a legislation of a country. Input should include 'question' and single 'country' in a dict",
    )
]

print(search_legislation.name)
print(search_legislation.description)
print(search_legislation.args)

# search_legislation.invoke({"question": "in which circumstances personal data can be processed?", "country": "Brazil"})


# @tool
# def get_word_length(word: str) -> int:
#     """Returns the length of a word."""
#     return len(word)


# get_word_length.invoke("abc")
# tools = [get_word_length]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable legal expert, when you need to ",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

prompt = hub.pull("hwchase17/react-chat")

llm_with_tools = llm.bind_tools(tools)
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "what's the difference between chinese and brazilian data legislation? Only use a tool if needed, otherwise respond with Final Answer.the search should be done step by step, country by country.",
        # Notice that chat_history is a string, since this prompt is aimed at LLMs, not chat models
        "chat_history": "",
    }
)
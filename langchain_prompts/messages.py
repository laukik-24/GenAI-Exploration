from email import message
from unittest import result
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage 
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("MEGALLM_API_KEY"),
    model="openai-gpt-oss-20b"
)

message = [
    SystemMessage(content = 'You are a helpful assistant'),
    HumanMessage(content = 'Tell me about langchain')
]

res = llm.invoke(message)

message.append(AIMessage(content=res.content))

print(message)

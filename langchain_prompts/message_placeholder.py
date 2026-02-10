from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("MEGALLM_API_KEY"),
    model="openai-gpt-oss-20b"
)

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human' , '{query}')
])

chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
    


prompt = chat_template.invoke({'chat_history' : chat_history , 'query' :  'Where is my refund?'})

print(prompt)

# res = llm.invoke(prompt)

# print(res.content)

# print(chat_history)
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage 
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("MEGALLM_API_KEY"),
    model="openai-gpt-oss-20b"
)

chat_template = ChatPromptTemplate([
    ('system' ,'You are a helpful {domain} expert' ),
    ('human' , 'Explain in simple terms , what is {topic}')
])

prompt = chat_template.invoke({'domain': 'cricket' , 'topic':'Dusra'})

res = llm.invoke(prompt)

print(res.content)
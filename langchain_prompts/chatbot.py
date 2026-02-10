from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage 
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("MEGALLM_API_KEY"),
    model="openai-gpt-oss-20b"
)

chat_history = [
    SystemMessage(content = 'You are a helpful AI assistant')
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content = user_input))
    if user_input == "exit":
        break
    
    res = llm.invoke(chat_history)
    chat_history.append(AIMessage(content = res.content))
    
    print("AI: ",res.content)

print(chat_history)
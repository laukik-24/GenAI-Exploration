from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-6")

res = model.invoke("What is capital of India?")

print(res.content)

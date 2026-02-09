from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

res = model.invoke("What is capital of India?")

print(res.content)

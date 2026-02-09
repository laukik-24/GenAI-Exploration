from unittest import result
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

docs = [
    "Hello, how are you?",
    "I am fine, thank you!",
    "What is your name?",
    "My name is John Doe.",
    "What is your age?",
    "I am 20 years old.",
]

result = embedding_model.embed_documents(docs)

print(result)
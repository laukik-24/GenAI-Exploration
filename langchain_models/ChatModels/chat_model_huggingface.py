from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",   # better supported
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

res = model.invoke("What is capital of India?")
print(res.content)

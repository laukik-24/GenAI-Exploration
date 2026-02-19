# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

load_dotenv ()

# llm = HuggingFaceEndpoint(
#     # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
#     task="text-generation",
# )

# model = ChatHuggingFace(llm=llm)

llm = ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("MEGALLM_API_KEY"),
    model="openai-gpt-oss-20b"
)

#1st prompt --> detailed report
template1 = PromptTemplate(
    template="Write a detialed report on {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Write a 5 line summary on following text: \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | llm | parser | template2 | llm | parser

res = chain.invoke({"topic" : "Black Hole"})

print(res)
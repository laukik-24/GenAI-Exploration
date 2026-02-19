from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv ()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    # repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

#1st prompt --> detailed report
template1 = PromptTemplate(
    template="Write a detialed report on {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Write a 5 line summary on following text: \n {text}",
    input_variables=['text']
)

prompt1 = template1.invoke({"topic":'BlackHole'})

res1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text" : res1.content})

res2 = model.invoke(prompt2)

print(res2.content)


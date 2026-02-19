from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import os

load_dotenv ()

llm = HuggingFaceEndpoint(
    # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
    # repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# llm = ChatOpenAI(
#     base_url="https://ai.megallm.io/v1",
#     api_key=os.getenv("MEGALLM_API_KEY"),
#     model="openai-gpt-oss-20b"
# )

parser = JsonOutputParser()

template = PromptTemplate(
    template="give me name, age and city of a fictional person. \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

# prompt = template.format()

# res = llm.invoke(prompt)

# final_res = parser.parse(res.content)

chain = template | model | parser

result = chain.invoke({})

print(result)

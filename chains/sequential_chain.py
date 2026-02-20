from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
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

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

res = chain.invoke({'topic': 'Ramzan'})

print(res)

chain.get_graph().print_ascii()
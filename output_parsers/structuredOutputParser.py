from re import template
from xxlimited import Str
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
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

schema = [
    ResponseSchema(name = 'fact_1' , description='Fact 1 about the topic'),
    ResponseSchema(name = 'fact_2' , description='Fact 2 about the topic'),
    ResponseSchema(name = 'fact_3' , description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


chain = template | model | parser

res = chain.invoke({'topic' : 'black hole'})

print(res)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18 , description="Age of the person ")
    city: str = Field(description="Name of the city the person belings to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person\n{format_instructions}",
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

chain = template | model | parser

res = chain.invoke({"place": "Europe"})

print(res)

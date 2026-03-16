from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

load_dotenv ()

llm = ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("MEGALLM_API_KEY"),
    model="openai-gpt-oss-20b"
)

loader = TextLoader('Document_loaders/cricket.txt' , encoding='utf-8')

docs = loader.load()
parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a summary of the following poem: {poem}",
    input_variables=['poem']
)

print(type(docs))
print(len(docs))
print(docs[0])
 
chain = prompt | llm | parser

res = chain.invoke({'poem':docs[0].page_content})
print(res)
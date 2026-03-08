from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from langchain_core.runnables import RunnableSequence

load_dotenv()

llm = ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("MEGALLM_API_KEY"),
    model="openai-gpt-oss-20b"
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=[ 'topic']
)
prompt2  = PromptTemplate(
    template="Explain the folowing text: {text}",
    input_variables=[ 'text']
)

chain = RunnableSequence(prompt1, llm, parser , prompt2 , llm , parser)

print(chain.invoke({ 'topic':'AI'})) 
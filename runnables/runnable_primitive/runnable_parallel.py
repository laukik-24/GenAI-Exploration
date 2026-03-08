from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from langchain_core.runnables import  RunnableParallel

load_dotenv()

llm = ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("MEGALLM_API_KEY"),
    model="openai-gpt-oss-20b"
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="gearate a tweet about {topic}",
    input_variables=[ 'topic']
)
prompt2  = PromptTemplate(
    template="gearate a linkedin post about {topic}",
    input_variables=[ 'topic']
)

chain = RunnableParallel({
    'tweet': prompt1 | llm | parser,
    'linkedin': prompt2 | llm | parser
})

res=chain.invoke({ 'topic':'AI'})

print(res) 
# print(re÷s.linkedin) 
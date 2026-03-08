from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from langchain_core.runnables import RunnableSequence, RunnablePassthrough,RunnableParallel  

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

joke_gen_chain = RunnableSequence(prompt1, llm, parser )

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough() | joke_gen_chain ,
    'explanation': RunnableSequence(prompt2, llm, parser)
}) 

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

res = final_chain.invoke({ 'topic':'AI'})

print(res) 
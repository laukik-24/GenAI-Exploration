from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from langchain_core.runnables import RunnableSequence, RunnablePassthrough,RunnableParallel , RunnableLambda , RunnableBranch



load_dotenv()

def word_counter(text):
    return len(text.split())

llm = ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("MEGALLM_API_KEY"),
    model="openai-gpt-oss-20b"
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a detailed report about {topic}",
    input_variables=[ 'topic']
)

prompt2 = PromptTemplate(
    template="Write a summary of the following text: {text}",
    input_variables=[ 'text']
)

report_chain = prompt1 | llm | parser
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, prompt2 | llm | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain, branch_chain)

res = final_chain.invoke({ 'topic':'AI'})

print(res)
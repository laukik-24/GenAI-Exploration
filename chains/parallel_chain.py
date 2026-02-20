from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv ()

llm = HuggingFaceEndpoint(
    # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
    # repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("MEGALLM_API_KEY"),
    model="openai-gpt-oss-20b"
)

prompt1 = PromptTemplate(
    template='Generate a short and simple notes form the following text \n {text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template='generate 5 short question answers from following text \n {text}',
    input_variables=['text']
)
prompt3 = PromptTemplate(
    template='merge the provided notes and quiz in single document \n notes-> {notes} and quiz -> {quiz}',
    input_variables=['notes' , 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merging_chain = prompt3 | model2 | parser

final_chain =  parallel_chain | merging_chain

text = """
Artificial Intelligence (AI) is the simulation of human intelligence in machines 
that are programmed to think and learn. AI systems can perform tasks such as 
speech recognition, decision-making, visual perception, and language translation.

Machine Learning (ML) is a subset of AI that enables systems to learn from data 
without being explicitly programmed. Deep Learning, a branch of ML, uses neural 
networks with many layers to analyze complex patterns in large datasets.

AI is widely used in healthcare for disease diagnosis, in finance for fraud 
detection, and in transportation for autonomous vehicles. However, ethical 
concerns such as bias, privacy, and job displacement remain significant challenges.

The future of AI depends on responsible development, transparency, and 
collaboration between governments, companies, and researchers.
"""


res = final_chain.invoke({'text':text})

print(res)

final_chain.get_graph().print_ascii()
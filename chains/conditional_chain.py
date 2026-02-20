from email.policy import default
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal
from dotenv import load_dotenv
import os


load_dotenv ()

# llm = HuggingFaceEndpoint(
#     # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
#     # repo_id="google/gemma-2-2b-it",
#     task="text-generation",
# )

# model1 = ChatHuggingFace(llm=llm)

model = ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("MEGALLM_API_KEY"),
    model="openai-gpt-oss-20b"
)


class Feedback(BaseModel) : 
    sentiment : Literal['positive' , 'negative'] = Field(description='Give the sentiment of the feedback')

str_parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template='Clasiify the sentiment of the following feedback into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': pydantic_parser.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

classifier_chain= prompt1 | model | pydantic_parser

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive',prompt2 | model | str_parser),
    (lambda x:x.sentiment == 'negative',prompt3 | model | str_parser),
    RunnableLambda(lambda x:'Could not find the sentiment')
)

final_chain = classifier_chain | branch_chain

feed = '''I recently purchased the premium subscription plan, and while the initial setup 
process was smooth, I’ve been experiencing frequent crashes in the mobile app. 
The app logs me out randomly, and sometimes my saved preferences disappear. 
Customer support responded quickly, but the issue still hasn’t been fully resolved. 
Overall, I like the features and design, but the stability problems are frustrating 
and affect my daily workflow. I hope this gets fixed soon.'''


res = final_chain.invoke({'feedback': feed})

print(res)

final_chain.get_graph().print_ascii()
import random


class NakliLLm:
    def __init__(self) -> None:
        print('LLM created')
    
    def predict(self,prompt):
        response_list = ["Delhi is capital of India" , "IPL is cricket league" , "AI stands for Artificial Intelligence"]
        
        return {random.choice(response_list)}

llm = NakliLLm()
llm.predict()
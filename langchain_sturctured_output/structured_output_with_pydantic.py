from langchain_openai import ChatOpenAI
from typing import  Annotated ,Optional , Literal
from pydantic import BaseModel ,Field
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    base_url="https://ai.megallm.io/v1",
    api_key=os.getenv("MEGALLM_API_KEY"),
    model="openai-gpt-oss-20b"
)

#schema
class Review(BaseModel):
    
    key_themes : list[str] = Field(description="Write down all the key themes disscussed in the review in a list")
    summary: str = Field(description="A brief summary of the review") 
    sentiment:Literal["pos" , "neg"] = Field(description="Return sentiment of the review either Positive, Negative or Neutral" )
    pros:Optional[list[str]] = Field( default=None, description="Write down all the pros in a list") 
    cons:Optional[list[str]] = Field(default=None, description="Write down all the cons in a list")
    name:Optional[str] = Field(default=None, description="Write down name of receiver")
    
structured_model = llm.with_structured_output(Review)

# result1 = structured_model.invoke('''The hardware is great, but the software feels bloated.There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.''')

result2 = structured_model.invoke('''I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors 

Review by Laukik ''')

print(result2)
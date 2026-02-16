from typing import Optional
from pydantic import BaseModel , EmailStr , Field

class Student(BaseModel):
    name : str = 'laukik'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10 , description='A decimal value representing cgpa of the student')
    
new_stud = {'age':22 , 'email':'abc@abc.in' , 'cgpa':7}

student = Student(**new_stud)

print(student.model_dump_json())
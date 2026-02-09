from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Hello, how are you?"

docs = ["Hello, how are you?", "I am fine, thank you!", "What is your name?", "My name is John Doe.", "What is your age?", "I am 20 years old."]

result1 = embedding_model.embed_query(text)

result2 = embedding_model.embed_documents(docs)

print(len(result1))
print(len(result2))
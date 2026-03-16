
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('Document_loaders/dl-curriculum.pdf')

doc = loader.load()

print (doc[0].page_content)
print (doc[0].metadata)

print(len(doc))
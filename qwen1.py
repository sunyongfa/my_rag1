from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader,Docx2txtLoader,PyPDFLoader,UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time,os

DASHSCOPE_API_KEY = "sk-c4f6b84c10054d14a91dc57bb6e8afb5"

root_dir = "./dosc"

def extract_file_dirs(directory):
    dosc=[]
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                fp = os.path.join(root, file)
                try:
                    pyload=PyPDFLoader(fp)
                    dosc.extend(pyload.load())
                except Exception as e:
                    print(f"Error loading {file} files:{e}")
            elif file.endswith(".txt"):
                fp = os.path.join(root, file)
                try:
                    txtload=TextLoader(fp,encoding="utf-8")
                    dosc.extend(txtload.load())
                except Exception as e:
                    print(f"Error loading {file} files:{e}")

           elif file.endswith(".docx"):
                fp = os.path.join(root, file)
                try:
                    docsload=Docx2txtLoader(fp)
                    dosc.extend(docsload.load())
                except Exception as e:
                    print(f"Error loading {file} files:{e}")

          elif file.endswith(".xlsx"):
                fp = os.path.join(root, file)
                try:
                    xlsxspoad = UnstructuredExcelLoader(fp)
                    dosc.extend(xlsxspoad.load())
                except Exception as e:
                    print(f"Error loading {file} files:{e}")
            else:
                 print("Unsupported file type")
    return dosc

docs = extract_file_dirs(root_dir)

# 文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
split_docs = text_splitter.split_documents(docs)

embeddings = DashScopeEmbeddings(
        model="text-embedding-v1", dashscope_api_key=DASHSCOPE_API_KEY
    )

db = Chroma.from_documents(split_docs, embeddings,persist_directory="./chroma/news_test")
# 持久化
db.persist()

from langchain_community.vectorstores import Chroma,FAISS
#from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader,Docx2txtLoader,PyPDFLoader,UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time,os
from dashscope.api_entities.dashscope_response import Role
from dashscope import Generation

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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=20)
split_docs = text_splitter.split_documents(docs)

embeddings = DashScopeEmbeddings(
        model="text-embedding-v1", dashscope_api_key=DASHSCOPE_API_KEY
    )

db = FAISS.from_documents(split_docs, embeddings)

messages = []
while True:
    message = input('user:')
    similarDocs = db.similarity_search(message, k=5)
    summary_prompt = "\n\n".join([doc.page_content for doc in similarDocs])
    # print(summary_prompt+"\n")
    for doc in similarDocs:
        print(doc.page_content)
        print("****************************")
    send_message = f"请根据{summary_prompt}对{message}的问题进行回答,如果根据{summary_prompt}不能对{message}进行回答，请直接用大模型来回答。"
    messages.append({'role': Role.USER, 'content': send_message})
    whole_message = ''
    # 切换模型
    responses = Generation.call(Generation.Models.qwen_max, messages=messages, result_format='message', stream=True,
                                incremental_output=True)
    # responses = Generation.call(Generation.Models.qwen_turbo, messages=messages, result_format='message', stream=True, incremental_output=True)
    print(type(responses))
    print('system:', end='')
    for response in responses:
        whole_message += response.output.choices[0]['message']['content']
        print(response.output.choices[0]['message']['content'], end='')
    print()
    messages.append({'role': 'assistant', 'content': whole_message})



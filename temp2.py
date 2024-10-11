from langchain_community.vectorstores import Chroma,FAISS
#from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader,Docx2txtLoader,PyPDFLoader,UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time,os
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Tongyi
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


os.environ["OPENAI_API_KEY"] = "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"
os.environ["DASHSCOPE_API_KEY"]="sk-c4f6b84c10054d14a91dc57bb6e8afb5"
llm = Tongyi(streaming=True,model="qwen-turbo",temperature=0.1)
llm2 = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)

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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
documents = text_splitter.split_documents(docs)
embeddings = DashScopeEmbeddings(
        model="text-embedding-v1", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )

#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#ollama_embeddings = OllamaEmbeddings(model="qwen2")
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("./faiss_index1")

retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

template = """
请根据{context}对{question}的问题进行回答，
如果你不能从{context},回答{question}，
直接用大模型来回答就好了。
请用中文输出答案。
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    query = input('user:')
    similarDocs = vectorstore.similarity_search(query, k=5)
    #summary_prompt = "\n\n".join([doc.page_content for doc in similarDocs])
    #print(summary_prompt+"\n")
    for doc in similarDocs:
        print(doc.page_content)
        print("****************************")
    response = chain.invoke(query)
    print("RAG 输出结果:", response)



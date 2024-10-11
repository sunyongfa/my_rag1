import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader,Docx2txtLoader,PyPDFLoader,UnstructuredExcelLoader
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Tongyi

os.environ["DASHSCOPE_API_KEY"]="***"
llm = Tongyi( model="qwen-turbo",temperature=0.1)


BGE_MODEL_PATH = "D:/pyspace/BAAI/bge-large-zh-v1.5"
root_dir = "./dosc"

docs = extract_file_dirs(root_dir)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
documents = text_splitter.split_documents(docs)
huggingface_bge_embedding = HuggingFaceBgeEmbeddings(model_name=BGE_MODEL_PATH)
#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#ollama_embeddings = OllamaEmbeddings(model="qwen2")
vectorstore = FAISS.from_documents(documents, huggingface_bge_embedding)
vectorstore.save_local("./faiss_index2")

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

'''chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)'''


from langchain_core.runnables import RunnableParallel

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

while True:
    query = input('user:')
    similarDocs = vectorstore.similarity_search(query, k=5)
    for doc in similarDocs:
        print(doc.page_content)
        print("****************************")

    response=rag_chain_with_source.stream(query)
    for chunk in response:
        for key in chunk:
            if key=='answer':
                print(chunk[key], end="", flush=True)
    print()

'''while True:
    query = input('user:')
    similarDocs = vectorstore.similarity_search(query, k=5)
    # summary_prompt = "\n\n".join([doc.page_content for doc in similarDocs])
    # print(summary_prompt+"\n")
    for doc in similarDocs:
        print(doc.page_content)
        print("****************************")
    response = chain.invoke(query)
    print("RAG 输出结果:", response)'''



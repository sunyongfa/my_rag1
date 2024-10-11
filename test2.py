import os
from langchain.llms.base import LLM
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader,Docx2txtLoader,PyPDFLoader,UnstructuredExcelLoader
from langchain_community.llms import Tongyi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma,FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
os.environ["DASHSCOPE_API_KEY"]="sk-c4f6b84c10054d14a91dc57bb6e8afb5"
llm = Tongyi( model="qwen-max",temperature=0.1)
BGE_MODEL_PATH = "D:/pyspace/BAAI/bge-large-zh-v1.5"
root_dir = "./dosc"

'''docs = extract_file_dirs(root_dir)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
documents = text_splitter.split_documents(docs)
huggingface_bge_embedding = HuggingFaceBgeEmbeddings(model_name=BGE_MODEL_PATH)
vectordb = FAISS.from_documents(documents=documents, embedding=huggingface_bge_embedding)
vectordb.save_local("./faiss_index2")'''

huggingface_bge_embedding = HuggingFaceBgeEmbeddings(model_name=BGE_MODEL_PATH)
vectordb=FAISS.load_local("./faiss_index2",embeddings=huggingface_bge_embedding,allow_dangerous_deserialization=True)

def create_original_query(original_query):
    query = original_query["question"]
    qa_system_prompt = """
            You are an AI language model assistant. Your task is to generate three 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines."""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{question}"),
        ]
    )
    rag_chain = (
            qa_prompt
            | llm
            | StrOutputParser()
    )

    question_string = rag_chain.invoke(
        {"question": query}
    )

    lines_list = question_string.splitlines()
    queries = []
    queries = [query] + lines_list

    return queries

from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


cross_encoder = CrossEncoder('D:/pyspace/bge-reranker-large')

def create_documents(queries):
    retrieved_documents = []
    for i in queries:
        results = vectordb.as_retriever(search_kwargs={'k': 10}).get_relevant_documents(i)
        docString = [doc.page_content for doc in results]
        retrieved_documents.extend(docString)

    unique_a = []
    # If there is duplication documents for each query, make it unique
    for item in retrieved_documents:
        if item not in unique_a:
            unique_a.append(item)

    unique_documents = list(unique_a)
    print(len(unique_documents))
    pairs = []

    for doc in unique_documents:
        pairs.append([queries[0], doc])

    scores = cross_encoder.predict(pairs)

    final_queries = []
    for x in range(len(scores)):
        final_queries.append({"score": scores[x], "document": unique_documents[x]})

    # Rerank the documents, return top 5
    sorted_list = sorted(final_queries, key=lambda x: x["score"], reverse=True)
    first_five_elements = sorted_list[:11]
    return first_five_elements

def create_documentsNo_Rank(queries):
    retrieved_documents = []
    for i in queries:
        results = vectordb.as_retriever(search_kwargs={'k': 10}).get_relevant_documents(i)
        docString = [doc.page_content for doc in results]
        retrieved_documents.extend(docString)

    unique_a = []
    # If there is duplication documents for each query, make it unique
    for item in retrieved_documents:
        if item not in unique_a:
            unique_a.append(item)

    unique_documents = list(unique_a)
    print(len(unique_documents))

    return unique_documents

qa_system_prompt = """
        Assistant is a large language model trained by OpenAI. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        
        {context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{question}"),
    ]
)

def format(docs):
    doc_strings = [doc["document"] for doc in docs]
    for doc in doc_strings:
        print(doc)
        print("*************************")
    return "\n\n".join(doc_strings)

def formatNo_rank(docs):
    doc_strings = [doc for doc in docs]
    print(len(doc_strings))
    for doc in doc_strings:
        print(doc)
        print("*************************")
    return "\n\n".join(doc_strings)

chain = (
    # Prepare the context using below pipeline
    # Generate Queries -> Cross Encoding -> Rerank ->return context
    {"context": RunnableLambda(create_original_query)| RunnableLambda(create_documents) | RunnableLambda(format), "question": RunnablePassthrough()}
    | qa_prompt
    | llm
    | StrOutputParser()
)



while True:
    question = input('user:')
    result=chain.invoke({"question": question})
    print(result+"\n")
  

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader,Docx2txtLoader,PyPDFLoader,UnstructuredExcelLoader
from langchain_community.llms import QianfanLLMEndpoint
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Tongyi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma,FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os,logging
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter


os.environ["DASHSCOPE_API_KEY"]="***"
llm = Tongyi( model="qwen-turbo",temperature=0.1)
BGE_MODEL_PATH = "D:/pyspace/BAAI/bge-large-zh-v1.5"
root_dir = "./dosc"

docs = extract_file_dirs(root_dir)

'''text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
documents = text_splitter.split_documents(docs)
huggingface_bge_embedding = HuggingFaceBgeEmbeddings(model_name=BGE_MODEL_PATH)
vectordb = FAISS.from_documents(documents=documents, embedding=huggingface_bge_embedding)
vectordb.save_local("./faiss_index1")'''

huggingface_bge_embedding = HuggingFaceBgeEmbeddings(model_name=BGE_MODEL_PATH)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
vectordb=FAISS.load_local("./faiss_index1",embeddings=huggingface_bge_embedding,allow_dangerous_deserialization=True)

redundant_filter = EmbeddingsRedundantFilter(embeddings=huggingface_bge_embedding)
relevant_filter = EmbeddingsFilter(embeddings=huggingface_bge_embedding, similarity_threshold=0.3)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[text_splitter, redundant_filter, relevant_filter]
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=vectordb.as_retriever(search_kwargs={'k': 10})
)

#---------------------------Prepare Multi Query Retriever--------------------


retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(search_kwargs={'k': 10}), llm=llm
)

#----------------------Setup QnA----------------------------------------
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

qa_system_prompt = """
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \

        {context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{question}"),
    ]
)

def format_docs(docs):
    doc_strings = [doc.page_content for doc in docs]
    print(len(doc_strings))
    for doc in doc_strings:
        print(doc)
        print("*************************")
    return "\n\n".join(doc_strings)


def format_docs1(docs):
    #Called the reordering function in here
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    doc_strings = [doc.page_content for doc in reordered_docs]
    print(len(doc_strings))
    for doc in doc_strings:
        print(doc)
        print("*************************")
    return "\n\n".join(doc_strings)

rag_chain = (
    {"context": retriever_from_llm  | format_docs1, "question": RunnablePassthrough()}
    | qa_prompt
    | llm
    | StrOutputParser()
)

#logging.basicConfig(level=logging.INFO)
#logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)
while True:
    question = input('user:')
    result=rag_chain.invoke(question)
    print(result+"\n")



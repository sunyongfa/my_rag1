import os,requests, json
from langchain.llms.base import LLM
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain_community.llms import Tongyi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

endpoint = "https://ukl-openai.openai.azure.com/"
deployment = "ukl-chatgpt4o"
GPT4o_KEY="***"
api_version = '2024-02-15-preview'
GPT_name = "GPT4o"
GPT4o_ENDPOINT =  endpoint + "openai/deployments/" +deployment+"/chat/completions?api-version="+api_version
headers = {
    "Content-Type": "application/json",
    "api--key": GPT4o_KEY,
}
BGE_MODEL_PATH = "D:/pyspace/rag1/BAAI/bge-large-zh-v1.5"
huggingface_bge_embedding = HuggingFaceBgeEmbeddings(model_name=BGE_MODEL_PATH)
vectordb = FAISS.load_local("./faiss_index_gpt4O", embeddings=huggingface_bge_embedding,
                            allow_dangerous_deserialization=True)

def create_original_query(query):

    history = []
    history.append({"role":"system","content":"""You are an AI language model assistant. Your task is to generate three 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines ."""})

    history.append({'role': 'user', 'content': query})
    payload = {
        "messages": history,
        "max_tokens": 1000,
        "temperature": 0.7,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_p": 0.95,
        "stop": "null"

        }

    try:
        response = requests.post(GPT4o_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise exit()

    message = response.json()["choices"][0]["message"]["content"].strip()
    lines_list = message.splitlines()
    queries = [query] + lines_list
    return queries

cross_encoder = CrossEncoder('D:/pyspace/rag1/bge-reranker-large')
def create_documents(queries):
    retrieved_documents = []
    for i in queries:
        results = vectordb.as_retriever(search_kwargs={'k': 5}).get_relevant_documents(i)
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
    first_five_elements = sorted_list[:6]
    return first_five_elements

def create_documentsNo_Rank(queries):
    retrieved_documents = []
    for i in queries:
        results = vectordb.as_retriever(search_kwargs={'k': 5}).get_relevant_documents(i)
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

def format(docs):
    doc_strings = [doc.page_content for doc in docs]
    #doc_strings = [doc["document"] for doc in docs]
    '''for doc in doc_strings:
        print(doc)
        print("*************************")'''
    return "\n\n".join(doc_strings)


def formatNo_rank(docs):
    doc_strings = [doc for doc in docs]
    print(len(doc_strings))
    for doc in doc_strings:
        print(doc)
        print("*************************")
    return "\n\n".join(doc_strings)

def init_prompt():
    global conversation_history
    conversation_history = []
    conversation_history.append({"role":"system","content":"You are an AI assistant ."})

def get_response(prompt):
    whole_message = ''
    conversation_history.append({'role': 'user', 'content': prompt})
    data = {
        "model": "GPT4o",
        "messages": conversation_history,
        "stream": True
    }
    try:
        response = requests.post(GPT4o_ENDPOINT, headers=headers, data=json.dumps(data), stream=True)
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data_line = decoded_line[6:]
                    if data_line == '[DONE]':
                        break
                    try:
                        json_data = json.loads(data_line)
                        choices = json_data['choices'][0]
                        if 'delta' in choices:
                            chunk_message = choices['delta'].get('content', '')
                            print(chunk_message, end='', flush=True)
                            whole_message += chunk_message
                    except json.JSONDecodeError:
                        continue
        conversation_history.append({'role': 'assistant', 'content': whole_message})
    except requests.RequestException as e:
        raise exit()
init_prompt()
while True:
    question = input('user:')
    #result1=create_original_query(question)
    #result2=create_documents(result1)
    results = vectordb.as_retriever(search_kwargs={'k': 10}).get_relevant_documents(question)
    context=format(results)
    qa_system_prompt = f"请根据{context}对{question}的问题进行回答,如果根据{context},不能回答{question}，请直接利用大模型来回答。"
    get_response(qa_system_prompt)
    print("\n")

import os,requests, json
from langchain.llms.base import LLM
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain_community.llms import Tongyi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

endpoint = "https://ukl-openai.openai.azure.com/"
deployment = "ukl-chatgpt4o"
GPT4o_KEY="1ef39609cfae494b940a5023ea8241e4"
api_version = '2024-02-15-preview'
GPT_name = "GPT4o"
GPT4o_ENDPOINT =  endpoint + "openai/deployments/" +deployment+"/chat/completions?api-version="+api_version
headers = {
    "Content-Type": "application/json",
    "api-key": GPT4o_KEY,
}

BGE_MODEL_PATH = "D:/pyspace/rag1/BAAI/bge-large-zh-v1.5"
huggingface_bge_embedding = HuggingFaceBgeEmbeddings(model_name=BGE_MODEL_PATH)
vectordb = FAISS.load_local("./faiss_index_gpt4O", embeddings=huggingface_bge_embedding,
                            allow_dangerous_deserialization=True)

def format(docs):
    doc_strings = [doc.page_content for doc in docs]
    #doc_strings = [doc["document"] for doc in docs]
    '''for doc in doc_strings:
        print(doc)
        print("*************************")'''
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
    results = vectordb.as_retriever(search_kwargs={'k': 10}).get_relevant_documents(question)
    context=format(results)
    qa_system_prompt = (f'''请根据下文：
                        {context}
                        对
                        {question}
                        的问题进行回答,如果不能回答{question},请直接利用大模型来回答。''')
    get_response(qa_system_prompt)
    print("\n")


        



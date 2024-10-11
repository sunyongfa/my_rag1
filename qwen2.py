from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
#from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

messages = []
DASHSCOPE_API_KEY = "sk-c4f6b84c10054d14a91dc57bb6e8afb5"
embeddings = DashScopeEmbeddings(
        model="text-embedding-v1", dashscope_api_key=DASHSCOPE_API_KEY
    )
db = Chroma(persist_directory="./chroma/news_test", embedding_function=embeddings)

while True:
    message = input('user:')
    similarDocs = db.similarity_search(message, k=3)
    summary_prompt = "\n\n".join([doc.page_content for doc in similarDocs])
    print(summary_prompt)
    send_message = f"请根据{summary_prompt}对{message}的问题进行回答,如果根据{summary_prompt}不能对{message}进行回答，请直接用大模型来回答。"
    messages.append({'role': Role.USER, 'content': send_message})
    whole_message = ''
    # 切换模型
    responses = Generation.call(Generation.Models.qwen_turbo, messages=messages, result_format='message', stream=True,
                                incremental_output=True)
    # responses = Generation.call(Generation.Models.qwen_turbo, messages=messages, result_format='message', stream=True, incremental_output=True)
    print('system:', end='')
    for response in responses:
        whole_message += response.output.choices[0]['message']['content']
        print(response.output.choices[0]['message']['content'], end='')
    print()
    messages.append({'role': 'assistant

import requests
import time
import base64

endpoint = "https://ukl-openai.openai.azure.com/"
deployment = "ukl-chatgpt4o"
GPT4o_KEY="2ef39609cfae494b940a5023ea8241e4"
api_version = '2024-02-15-preview'
GPT_name = "GPT4o"
GPT4o_ENDPOINT =  endpoint + "openai/deployments/" +deployment+"/chat/completions?api-version="+api_version
headers = {
    "Content-Type": "application/json",
    "api-key": GPT4o_KEY,
}

def add_image_to_message(IMAGE_PATH):
    encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
    # 生成包含图片的prompt
    message = {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_image}"
          }
        },
        {
          "type": "text",
          "text": "Please find information about the picture." 
        }
      ]
    }
    return message

def init_prompt():
    global conversation_history
    conversation_history = []
    conversation_history.append({"role":"system","content":"""You are an AI language model assistant. Your task is to generate three 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines ."""})
def get_response(prompt):
    conversation_history.append({'role': 'user', 'content': prompt})
    # Send request
    payload = {
        "messages": conversation_history,
        "max_tokens": 1000,
        "temperature": 0.7,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_p": 0.95,
        "stop": "null"

        }
    try:
        response = requests.post(GPT4o_ENDPOINT, headers=headers, json=payload,stream=True)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise exit()

    # Handle the response as needed (e.g., print or process)
    # print(response.json())
    message = response.json()["choices"][0]["message"]["content"].strip()
    return message

print("Welcome to the "+GPT_name+" "+api_version+ "chatbot!")
init_prompt()

while True:
    # Get user input
    user_input = input("You:").strip()
    if user_input == "#e":
        break
    # Send user input to the CHATGPT model and get a response
    start_time = time.time()
    response = get_response(user_input)
    end_time = time.time()
    process_time = round(end_time - start_time, 2)
    # Print the response
    lines_list = response.splitlines()
    print(lines_list)
    print(f"GPT4o: {response} ({process_time}s)")
    conversation_history.append({'role': 'assistant', 'content': response})

    # Wait for a moment before continuing
    time.sleep(1)
print("Bye!")



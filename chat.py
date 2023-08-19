from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import json 
import numpy as np
from fastapi import FastAPI, Request, Form
import random
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
import openai
import os 
from fastapi.staticfiles import StaticFiles
from langchain.callbacks import get_openai_callback
import re
import time
import sys
import json
import random
import csv
from fastapi.responses import FileResponse
import requests
from fastapi.middleware.cors import CORSMiddleware


# List of allowed origins (origins that can make cross-origin requests)
origins = [
    "http://localhost:8000",
    # Add more origins if needed
]

# Step 1: Import the logging module
import logging
# Step 2: Set up a logger instance
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('logs.log')
# Step 3: Configure logging level and output format
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)
log_format = '%(asctime)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Function to load a dictionary from a JSON file
def load_dict_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to save a dictionary to a JSON file
def save_dict_to_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)
#open_ai_model
temp='%s%k%-%X%K%H%Y%C%v%j%K%d%k%T%d%g%T%t%z%J%I%Y%d%T%3%B%l%b%k%F%J%I%O%K%x%p%h%H%6%9%v%r%I%x%S%e%G%W%6%K%T'
api_key=""
for i in range(1,len(temp),2):
    api_key+=temp[i]
os.environ["OPENAI_API_KEY"] =api_key
COMPLETIONS_MODEL="text-davinci-003"
def Zbot(prompt,COMPLETIONS_MODEL,temperature):
        bot_response = openai.Completion.create(
            prompt=prompt,
            temperature=1,
            max_tokens=700,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=COMPLETIONS_MODEL
        )["choices"][0]["text"].strip(" \n")
        return bot_response

def convert_to_short_parts(response, max_length):
        parts = []
        pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(?<!\d\.)\s"
        sentences = re.split(pattern, response)
        current_part = ""
        for sentence in sentences:
            if len(current_part) + len(sentence) <= max_length:
                current_part += sentence
            elif sentence.endswith('.'):
                current_part += sentence
                parts.append(current_part)
                current_part = ""
            else:
                parts.append(current_part)
                current_part = sentence
        if current_part != '':
            parts.append(current_part)
        parts = list(filter(lambda item: item != '', parts))
        return parts

def edit_sentences(sentences):
            def is_emoji(character):
                ascii_value = ord(character)
                return 1000 <= ascii_value  # نطاق السمايلات في ترميز ASCII

            result = []
            previous_sentence = ""

            for s in range(len(sentences)):
                temp=""
                for i in range(len(sentences[s])):
                    if is_emoji(sentences[s][i]):
                        temp+=sentences[s][i]
                    else:
                        break
                if temp!="":
                    sentences[s-1]=sentences[s-1]+temp
                    sentences[s]=sentences[s][len(temp):]
            sentences = list(filter(lambda item: item != '', sentences))         
            return sentences

def conversation(user_response):
    user_response,user_id=user_response.split('-#-')
    data = load_dict_from_json('data.json')
    user=data[user_id]
    if user_response.strip().upper()=="MEMORIZE":
        prompt="""
            Vocabularies: {}
            explain all Vocabularies like dectionary.⚙️🤖💬
            use many and suitable emojis.
            you must to explain each vocabulary or word as following template:
            word:[]
            word type:[]
            definition:[] use emojis please.
            Common synonyms:[]
            
            
            """.format(user['Vocabularies'])
        Z=Zbot(prompt,COMPLETIONS_MODEL,1)
        Z=Z.split('\n')
        Z=[x for x in Z if len(x.strip())>0 ]
        return Z
    if user['step'] == 'step1':
        user['Vocabularies']=user_response.split(',')
        data[user_id]=user
        prompt="""
        please use [] in response.
        use the following vocabularies {} to create  3 role-playing  scenario.
        1:
        "user role":["Person Character"]
        "bot role":["Person Character"]
        "scenario":["description"]
        2:
        "user role":["Person Character"]
        "bot role":["Person Character"]
        "scenario":["description"]
        3:
        "user role":["Person Character"]
        "bot role":["Person Character"]
        "scenario":["description"]
        """.format(user['Vocabularies'])
        Z=Zbot(prompt,COMPLETIONS_MODEL,1)
        while True:
            if Z[-1]==']':
                break
            else:
                Z=Zbot(prompt,COMPLETIONS_MODEL,1)

        option1,option2,option3=Z[Z.find('1')+2:Z.find('2')],Z[Z.find('2')+2:Z.find('3')],Z[Z.find('3')+2:]
        user['Role_Play_Options']=[option1,option2,option3]
        user['step']='step2'
        data[user_id]=user
        save_dict_to_json(data, 'data.json')
        option=user['Role_Play_Options'][user['option']]
        p1=option[:option.find('bot role')-1]
        p2=option[option.find('bot role')-1:option.find('scenario')-1]
        p3=option[option.find('scenario')-1:]
        return [p1,p2,p3,"Do you want to start the following role play scenario?\n1- Yes\n2- No,I want another scenario"]
    if user['step'] == 'step2':
        if user_response.lower().strip()=='yes':
            user['step']='step3'
            user['prompt']="""
                    Act as "Bot role" to start our conversation to learn me the following Vocabularies.use many emojis.
                    first introuduce yourself.
                    Just return Bot response.
                    let's make our conversation shortly with many Emojis.
                    "vocabularies":{}
                    {}
                    Bot :
                    \n""".format(user['Vocabularies'],user['Role_Play_Options'][user['option']])
            Z=Zbot(user['prompt'],COMPLETIONS_MODEL,1)
            user['prompt']+=Z
            data[user_id]=user
            save_dict_to_json(data, 'data.json')
            return [Z]
        else:
            user['option']=(user['option']+1)%len(user['Role_Play_Options'])
            data[user_id]=user
            save_dict_to_json(data, 'data.json')
            option=user['Role_Play_Options'][user['option']]
            p1=option[:option.find('bot role')-1]
            p2=option[option.find('bot role')-1:option.find('scenario')-1]
            p3=option[option.find('scenario')-1:]
            return [p1,p2,p3,"Do you want to start the following role play scenario?\n1- Yes\n2- No,I want another scenario"]
            
    if user['step'] == 'step3':
        user['prompt']+='\n'+'User:'+user_response+'\nBot:'
        Z=Zbot(user['prompt'],COMPLETIONS_MODEL,1)
        user['prompt']+=Z
        data[user_id]=user
        save_dict_to_json(data, 'data.json')
        edit_result = convert_to_short_parts(Z, 30)
        edit_result = edit_sentences(edit_result)
        return edit_result


    

script_dir = os.path.dirname(__file__)
st_abs_file_path = os.path.join(script_dir, "static/")

app = FastAPI()


templates = Jinja2Templates(directory="")
app.mount("/static", StaticFiles(directory=st_abs_file_path), name="static")

# Enable CORS for all routes with allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    try:
        # Logging an informational message
        logger.info("Received a GET request to /")
        user = {
            'Vocabularies':None,
            'Role_Play_Options':None,
            'option':0,
            'prompt':'',
            'step':'step1'
            
        
    }
        

        username =random.randint(1,999999999)
        data = load_dict_from_json('data.json')
        user['start_time']=time.time()
        data[username]=user
        save_dict_to_json(data, 'data.json')
        return templates.TemplateResponse("home.html", {"request": request, "username": username})
    except Exception as e:
        # Logging an error message
        logger.error(f"An error occurred: {str(e)}")
        # Handle the exception and return an appropriate response
@app.get("/getChatBotResponse")
def get_bot_response(msg: str,request: Request):
    try:
        # Logging an informational message
        logger.info("Received a GET request to /getChatBotResponse")
        result = conversation(msg)
        return result
    except Exception as e:
        # Logging an error message
        logger.error(f"An error occurred: {str(e)}")
        # Handle the exception and return an appropriate response


if __name__ == "__main__":
    uvicorn.run("chat:app", reload=True)

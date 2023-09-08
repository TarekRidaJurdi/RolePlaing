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
openai.api_key =api_key
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

def grammar_check(text):
    # إعداد النص للتحقق من الأخطاء القواعدية
    payload = {'language': 'en-US', 'text': text}

    # إرسال طلب التحقق إلى خدمة LanguageTool
    response = requests.post('https://api.languagetool.org/v2/check', data=payload)

    # تحليل الاستجابة
    errors = []
    if response.status_code == 200:
        data = response.json()
        for mistake in data['matches']:
            if mistake['message']not in ['This sentence does not start with an uppercase letter.','Possible typo: you repeated a whitespace']:
                errors.append(mistake['message'])

    return errors
def ZbotChecker(text):
    errors=grammar_check(text)
    if errors:
        prompt="Correct “Text:{}” to standard English and place the results in “Correct Text:”".format(text)
        return Zbot(prompt,"text-davinci-003",1)
    else:
        return False

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
        data[user_id]=user
        user['step']='step2'
        data[user_id]=user
        save_dict_to_json(data, 'data.json')
        return ["Let's start learning Vocabularies with our Role-play 🎭👑🎲👗🎬","I will create 3 scenarios based on the vocabularies you want to learn.1️⃣2️⃣3️⃣","type anything to continue...⚙️🤖💬"]
    
    if user['step'] == 'step2':
            data[user_id]=user
            user['step']='step3'
            data[user_id]=user
            save_dict_to_json(data, 'data.json')
            return ["please Write the Vocabularies you want to learn in this format: word1, word2, word3","📚📖🔤📝🗣️"]




    if user['step'] == 'step3':
        user['Vocabularies']=user_response.split(',')
        data[user_id]=user
        prompt="""
        please use [] in response.
        use the following vocabularies {} to create  3 role-playing  scenario.
        The scenario must be specific and similar to a realistic situation.
        use emojis related to each scenario.
        1:
        "user role":["Person Character"]
        "bot role":["Person Character"]
        "scenario":["description for story"]
        2:
        "user role":["Person Character"]
        "bot role":["Person Character"]
        "scenario":["description for story"]
        3:
        "user role":["Person Character"]
        "bot role":["Person Character"]
        "scenario":["description for story"]
        """.format(user['Vocabularies'])
        Z=Zbot(prompt,COMPLETIONS_MODEL,1)
        while True:
            if Z[-1]==']':
                break
            else:
                Z=Zbot(prompt,COMPLETIONS_MODEL,1)

        option1,option2,option3=Z[Z.find('1')+2:Z.find('2')],Z[Z.find('2')+2:Z.find('3')],Z[Z.find('3')+2:]
        user['Role_Play_Options']=[option1,option2,option3]
        user['step']='step4'
        data[user_id]=user
        save_dict_to_json(data, 'data.json')
        option=user['Role_Play_Options'][user['option']]
        p1=option[:option.find('bot role')-1]
        p2=option[option.find('bot role')-1:option.find('scenario')-1]
        p3=option[option.find('scenario')-1:]
        return [p1,p2,p3,"Do you want to start the following role play scenario?","type 1️⃣ to start the scenario","type 0️⃣ to change the scenario"]
   
    if user['step'] == 'step4':
        if user_response.lower().strip()=='1':
            user['step']='step5'
            user['prompt']="""
                    Act as "Bot role" to start our conversation to learn me the following Vocabularies.use Emojis please.
                    first introuduce yourself.
                    You must not to get out of scenario.
                    Remind everything the user say. 
                    Just return Bot response.
                    let's make our conversation shortly.

                    "vocabularies":{}
                    {}
                    history:
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
            return [p1,p2,p3,"Do you want to start the following role play scenario?","type 1️⃣ to start the scenario","type 0️⃣ to change the scenario"]
            
    if user['step'] == 'step5':
        user['prompt']+='\n'+'User:'+user_response+'\nBot:'
        Z=Zbot(user['prompt'],COMPLETIONS_MODEL,1)
        user['prompt']+=Z
        user['correct']=False
        correct=ZbotChecker(user_response)
        if correct:
            user['correct']=correct.replace('Correct','Corrected')
        edit_result = convert_to_short_parts(Z, 30)
        edit_result = edit_sentences(edit_result)
        if user['correct']:
            a='<span style="color: green;">'+user['correct']+'✏️📝🔍📚📖'+'</span>'
            edit_result.insert(0,a)
        data[user_id]=user
        save_dict_to_json(data, 'data.json')
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
            'step':'step1',
            "correct":False
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
        return [str(e)]
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
        return [str(e)]

if __name__ == "__main__":
    uvicorn.run("chat:app", reload=True)

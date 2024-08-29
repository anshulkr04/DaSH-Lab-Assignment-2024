from datetime import datetime
import random as random
import requests
import urllib.parse
import json

now = datetime.now()

clientID = int(now.timestamp() * 1000)

def get_time():
	now = datetime.now()

	current_time_milliseconds = int(now.timestamp() * 1000)
	return current_time_milliseconds

def send_request(prompt , clientID):
    prompt = urllib.parse.quote(prompt)
    url = f'http://127.0.0.1:8000/userPrompt/{clientID}/{prompt}'
    response = requests.post(url)
    return response
final_data = []
with open("inputs.txt", "r") as file:
    lines = file.readlines()
    line_index = [random.randint(0, len(lines) - 1) for _ in range(3)] 

    for i in line_index:
        prompt = lines[i]
        timeSent = get_time() 
        res = send_request(prompt, clientID=clientID) 
        response = res.text 
        timeRecieved = get_time()  
        data = {
            "Prompt": prompt.strip(), 
            "Message": response,
            "ClientID": clientID,
            "Time Sent": timeSent,
            "Time Recieved": timeRecieved,
            "Source": "Groq"
        }
        final_data.append(data)
    with open(f'output-{clientID}.json' , "w") as output:
         json.dump(final_data , output , indent=4)



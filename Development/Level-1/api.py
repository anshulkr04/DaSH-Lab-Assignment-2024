from groq import Groq
import json
from datetime import datetime

def get_time():
	now = datetime.now()
	current_time_milliseconds = int(now.timestamp() * 1000)
	return current_time_milliseconds


client = Groq()
def execute(prompt):
	
	completion = client.chat.completions.create(
		model="llama3-8b-8192",
		messages=[
			{
				"role": "user",
				"content": f'explain in 20 words. {prompt}'
			}
		],
		temperature=0.7,
		max_tokens=100,
		top_p=1,
		stream=True,
		stop=None,
	)
	response = ""
	for chunk in completion:
		response += chunk.choices[0].delta.content or ""
	return response

def final_func():
	data = []
	with open("inputs.txt" , "r") as file:
		while file:
			line = file.readline()
			if line =="":
				break
			timeSent = get_time()	
			ans = execute(line)
			timeRecieved = get_time()
			data.append({
				"Prompt" : line,
				"Message": ans,
				"Time Sent": timeSent,
				"Time Recieved": timeRecieved,
				"Source": "Groq"
			})
	with open('output.json' , 'w') as json_file:
		json.dump(data, json_file, indent=4)

final_func()




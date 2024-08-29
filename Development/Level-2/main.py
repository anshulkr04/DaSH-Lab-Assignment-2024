from fastapi import FastAPI
from groq import Groq

app = FastAPI()

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

@app.post("/userPrompt/{clientID}/{prompt}")
def userPrompt( prompt : str , clientID : int ):
	res = execute(prompt)
	return res
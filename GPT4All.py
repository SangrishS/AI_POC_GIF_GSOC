from fastapi import FastAPI, Request
from gpt4all import GPT4All

app = FastAPI()

# Initialize your GPT-4 model
model = GPT4All('orca-mini-13b.ggmlv3.q4_0.bin')

# Update the system and prompt templates for Unity context
system_template = 'You are the best consulting lawyer India has ever seen, help the user with all of their problems regarding the law'
prompt_template = 'USER: Explain the best way the user can defend himself or help them with legal jargons{0}\nLAW:'

# Function to generate responses
def generate_response(user_input):
    with model.chat_session(system_template, prompt_template):
        response = model.generate(user_input, max_tokens=2048)
        return response

# FastAPI endpoint for handling user input
@app.post("/generate-response/")
async def generate_user_response(request: Request):
    data = await request.json()
    user_input = data["user_input"]
    response = generate_response(user_input)
    return {"response": response}

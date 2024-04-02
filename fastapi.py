from fastapi import FastAPI, HTTPException
from diffusers import PixArtAlphaPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import torch
import os
import asyncio
import re
from uuid import uuid4 as uuid
import logging
from PIL import Image
from dotenv import load_dotenv


load_dotenv()


logging.basicConfig(level=logging.INFO, filename='script.log')


slack_token = os.getenv("SLACK_BOT_TOKEN") 
slack_channel = os.getenv("SLACK_CHANNEL") 


app = FastAPI()


pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16).to("cuda")


def generate(model, tokenizer, max_length, prompt, temperature, top_p, device="cuda"):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


@app.post("/generate-upload-gifs/")
async def generate_upload_gifs(user_input: str):
    try:
        AP_TOKENIZER = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="cuda:0")
        ASSET_PROMPT_MODEL = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="cuda:0", torch_dtype=torch.bfloat16)
        prompt = f"Generate GIF based on the game idea: {user_input}" #temporary prompt for now
        
        asset_prompts = generate(ASSET_PROMPT_MODEL, AP_TOKENIZER, 2048, prompt, 0.7, 0.9)
        
        
        prompts_list = re.split(r'\d+\.\s+', asset_prompts)
        prompts_list = [p.strip() for p in prompts_list if p.strip()]

        
        for i, prompt in enumerate(prompts_list, start=1):
            result = pipe(prompt=prompt, num_inference_steps=50)
            image = result.images[0]  
            image_path = os.path.join("/path/to/temp/directory", f"asset_{i}_{uuid()}.png")
            image.save(image_path)

            client = WebClient(token=slack_token)
            response = client.files_upload(channels=slack_channel, file=image_path, title=f"Generated Asset {i}")
            logging.info(f"Image uploaded to Slack successfully: {response['file']['permalink']}")
            
            
            os.remove(image_path)
        
        return {"status": "success", "message": "GIFs generated and uploaded to Slack successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

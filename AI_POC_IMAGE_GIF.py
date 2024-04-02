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
import scipy.io.wavfile
import shutil
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, filename='script.log')

# Slack setup
slack_token = "secret" # Assuming you have set your token in .env
client = WebClient(token=slack_token)
slack_channel = "secret"
as_prompt = """Instruction:You are a GIF based prompt generator for creating game assets based on Input and #Assets:

Make it exactly what the User wants
Here are references of the prompts you should generate only which is required in #Assets Need fill in the details under []. Give it in a list and only for the assets required by the game in #Assets assetLoader:
Examples for how each type needs to be made, these are references only and should only be made for #Assets:
* Image-1: “Generate a pixelated character for the main hero, capturing the essence of [game theme]. This leading character should be bold and iconic, designed with a singular, vibrant texture. Keep it 2D, pixel-perfect, without shadows, reflecting the hero's unique traits in no more than 10 words. Name the asset char_hero.”
* Image_2: “Create a pixelated representation of  main character, themed around [game theme]. The character should stand out with a menacing yet simplistic design, using a single, striking texture. Ensure it's 2D, in-game appropriate, shadow-free, and described concisely in 10 words. The asset should be labeled char_char.”
* Image_3: “Single pixelated game asset.Create a detailed 2D game asset for a game [player/asset]. The character should be a single texture, in-game,pixelated, and no shadows. [The character should be described in 10 words]. The character should be named char_player.”
* Image_4: “Single pixelated game asset.Create a detailed 2D game asset for a game [enemy/asset]. The villain should be a single texture, in-game,pixelated, and no shadows. [The villain should be described in 10 words]. The villain should be named char_enemy.”
* image_5: “Generate a pixelated character for the main hero, capturing the essence of [game theme]. This leading character should be bold and iconic, designed with a singular, vibrant texture. Keep it 2D, pixel-perfect, without shadows, reflecting the hero's unique traits in no more than 10 words. Name the asset char_hero.”
* image_6: “Generate a pixelated character for the main hero, capturing the essence of [game theme]. This leading character should be bold and iconic, designed with a singular, vibrant texture. Keep it 2D, pixel-perfect, without shadows, reflecting the hero's unique traits in no more than 10 words. Name the asset char_hero.”
* image_7: “Generate a pixelated character for the main hero, capturing the essence of [game theme]. This leading character should be bold and iconic, designed with a singular, vibrant texture. Keep it 2D, pixel-perfect, without shadows, reflecting the hero's unique traits in no more than 10 words. Name the asset char_hero.”
* image_8: “Single pixelated game asset.Create a detailed 2D game asset for a game sub-asset. The sub-asset should be a single texture, in-game,pixelated, and no shadows. [The sub-asset should be described in 10 words]. The sub-asset should be named others.”
* image_9: “Generate a pixelated character for the main hero, capturing the essence of [game theme]. This leading character should be bold and iconic, designed with a singular, vibrant texture. Keep it 2D, pixel-perfect, without shadows, reflecting the hero's unique traits in no more than 10 words. Name the asset char_hero.”
Only send the prompts in a numbered list
"""
# Initialize the image generation pipeline
pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16).to("cuda")

# Function to generate text prompts for assets
def generate(model, tokenizer, max_length, prompt, temperature, top_p, device="cuda"):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, temperature=temperature, top_p=top_p, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

async def generate_assets_prompt(data: str):
    AP_TOKENIZER = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="cuda:0")
    ASSET_PROMPT_MODEL = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="cuda:0", torch_dtype=torch.bfloat16)
    prompt = f"Instruction to generate text prompts for assets based on the GIF idea: {data}. Use this rule if needed to generate{as_prompt}"
    output = generate(ASSET_PROMPT_MODEL, AP_TOKENIZER, 2048, prompt, 0.7, 0.9)
    return output

async def generate_and_upload_assets(prompts: str):
    tmp_dir = "path/to/temp/directory"
    os.makedirs(tmp_dir, exist_ok=True)
    
    prompts_list = re.split(r'\d+\.\s+', prompts)
    prompts_list = [p.strip() for p in prompts_list if p.strip()]

    for i, prompt in enumerate(prompts_list, start=1):
        result = pipe(prompt=prompt, num_inference_steps=50)
        image = result.images[0]  # Assuming each prompt generates one image
        image_path = os.path.join(tmp_dir, f"asset_{i}_{uuid()}.png")
        image.save(image_path)

        try:
            response = client.files_upload(channels=slack_channel, file=image_path, title=f"Generated Asset {i}")
            logging.info(f"Image uploaded to Slack successfully: {response['file']['permalink']}")
        except SlackApiError as e:
            logging.error(f"Error uploading file: {e}")
        
        # Clean up by removing the image file after upload
        os.remove(image_path)

# Example usage
async def main():
    game_idea = input("Enter user prompt here:")
    asset_prompts = await generate_assets_prompt(game_idea)
    await generate_and_upload_assets(asset_prompts)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())

#Recomend to run this on Google colab with t4 GPu
!pip install --upgrade diffusers accelerate transformers

from diffusers import DiffusionPipeline
import torch

from google.colab import userdata
userdata.get('hugging_face')

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
prompt = "create a futuristic drawing of a cat on the moon"
pipline = pipeline.to("cuda")
generator = torch.Generator("cuda").manual_seed(0)
image = pipeline(prompt, generator= generator).images[0]
image

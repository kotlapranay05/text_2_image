!pip install flask diffusers transformers torch torchvision Pillow flask-ngrok

from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
from PIL import Image

# Set the model and device
modelid = "stabilityai/stable-diffusion-2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model with correct precision based on the device
pipe = StableDiffusionPipeline.from_pretrained(
    modelid,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe.to(device)

# Check if model is loaded correctly
print(f"Model loaded on {device} device.")

# Set a simple and explicit prompt
prompt = input()


# Generate the image (avoid autocast if on CPU)
try:
    result = pipe(prompt, guidance_scale=8.5, height=512, width=512)
    # Check if the image is valid
    if result.images[0] is not None:
        result.images[0].save("generated_image.png")
        print("Image saved successfully!")
    else:
        print("Error: Generated image is empty.")
except Exception as e:
    print(f"Error during image generation: {e}")

from google.colab import files
files.download("generated_image.png")

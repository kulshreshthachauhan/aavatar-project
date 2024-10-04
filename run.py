import argparse
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

# Function to generate image based on the input image and text prompt
def generate_image(image_path, text_prompt, output_path):
    # Load the pre-trained model from HuggingFace
    pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Load the input image
    init_image = Image.open(image_path).convert("RGB")
    
    # Generate the output image based on text prompt
    generated_image = pipe(prompt=text_prompt, image=init_image).images[0]
    
    # Save the generated image
    generated_image.save(output_path)
    print(f"Generated image saved at: {output_path}")

# Main function to handle command line arguments
def main():
    parser = argparse.ArgumentParser(description="Generate image based on input image and text prompt")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image")
    parser.add_argument('--text-prompt', type=str, required=True, help="Text prompt for generating the scene")
    parser.add_argument('--output', type=str, required=True, help="Path to save the generated image")
    
    args = parser.parse_args()

    # Generate the image based on input arguments
    generate_image(args.image, args.text_prompt, args.output)

if _name_ == "_main_":
    main()

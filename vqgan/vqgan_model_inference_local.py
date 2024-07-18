import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

# Load VQGAN configuration and weights
config_path = "configs/model.yaml"  # Path to the config file
model_path = "ckpts/last.ckpt"  # Path to the pre-trained model checkpoint

config = OmegaConf.load(config_path)
model = VQModel(**config.model.params)

state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
model.load_state_dict(state_dict, strict=False)
model.eval()

# Function to preprocess the image
def preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    #image = Image.open(image_path).convert('RGB')
    image = Image.open(image_path)
    image2 = image
    b, g, r = image.split()
    image = Image.merge("RGB", (r, g, b))
    image2 = image
    image = transform(image).unsqueeze(0)
    return image, image2

# Function to convert tensor to image and display it using OpenCV
def tensor_to_image(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)  # Remove the batch dimension
    image = image.permute(1, 2, 0)  # Change dimensions to (H, W, C)
    image = image * 0.5 + 0.5  # Denormalize
    image = image.clamp(0, 1)  # Ensure pixel values are between 0 and 1
    image = (image.numpy() * 255).astype(np.uint8)  # Convert to uint8
    return image

# Example usage
image_path = "frame.jpg"
image_path = "data/45.jpg"
image_size = (160, 256)  # Adjust size as needed

# Preprocess the image
image, temp1 = preprocess_image(image_path, image_size)

# Encode and decode the image
with torch.no_grad():
    #_, _, _, recon_image = model(image)
    recon_image, s = model(image)

# Convert tensors to images
#original_image = tensor_to_image(image)
reconstructed_image = tensor_to_image(recon_image)

# Display the original and reconstructed images using OpenCV
cv2.imshow('Original Image', np.array(temp1)[:, :, ::-1])
cv2.imshow('Reconstructed Image', reconstructed_image)
cv2.waitKey(0)  # Wait for a key press to close the windows
cv2.destroyAllWindows()

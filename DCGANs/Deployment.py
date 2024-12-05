import torch
import torch.nn as nn
from torchvision import transforms
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3  # For RGB images
Z_DIM = 100  # Latent vector size
NUM_EPOCHS = 100
FEATURES_DISC = 64
FEATURES_GEN = 64  # Features for the Generator

# Define the Generator architecture
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

# Initialize and load the model
generator = Generator(z_dim=Z_DIM, channels_img=CHANNELS_IMG, features_g=FEATURES_GEN)
generator.load_state_dict(torch.load("generator.pth", map_location=torch.device('cpu')))
generator.eval()

# Function to generate an image
def generate_image():
    noise = torch.randn(1, Z_DIM, 1, 1)  # Latent vector reshaped to match expected input
    with torch.no_grad():
        generated_img = generator(noise)  # Generate an image

    # Convert tensor to PIL image for Streamlit
    generated_img = generated_img.squeeze().cpu().detach()
    generated_img = transforms.ToPILImage()(generated_img)
    return generated_img

# Streamlit app to generate and display the image
def main():
    st.title("DCGAN Image Generation")
    
    if st.button("Generate Image"):
        generated_img = generate_image()
        st.image(generated_img, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()

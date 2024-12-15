import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

# Check device (GPU/CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", device=device)

# Set the root directory for your dataset
root_dir = "/mnt/nas/dataset/BKMS2/part2017-2019"
output_dir = "/mnt/nas/dataset/BKMS2/clip_embeddings"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each year folder (2017, 2018, 2019)
for year in os.listdir(root_dir):
    year_path = os.path.join(root_dir, year)
    if os.path.isdir(year_path):
        print(f"Processing folder: {year}")
        
        # Prepare output file for the embeddings
        embeddings_output = os.path.join(output_dir, f"{year}_embeddings.pt")
        embeddings_dict = {}
        
        # Process each image file in the folder
        for img_file in tqdm(os.listdir(year_path)):
            try:
                # Load and preprocess the image
                img_path = os.path.join(year_path, img_file)
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                
                # Generate image embeddings
                with torch.no_grad():
                    image_features = model.encode_image(image).cpu()
                
                # Store the image features in the dictionary
                embeddings_dict[img_file] = image_features.squeeze(0)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        # Save embeddings dictionary for the year
        torch.save(embeddings_dict, embeddings_output)
        print(f"Saved embeddings for {year} to {embeddings_output}")

print("All embeddings have been processed and saved.")
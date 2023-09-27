import os
import requests
from tqdm import tqdm


def main(
    ckpt_url,
    save_dir='./checkpoints/pretrained',
):
    filename = os.path.basename(ckpt_url)
    # Define a filename for the downloaded weights
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        print(f'The checkpoint file {save_path} already exists, aborting...')
        return

    if not os.path.exists(save_dir):
        # Create the output directory if necessary
        os.makedirs(save_dir, mode=0o777, exist_ok=True)

    # Send an HTTP GET request to download the file
    response = requests.get(ckpt_url, stream=True)

    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        # Get the total file size in bytes from the response headers
        total_size = int(response.headers.get('content-length', 0))

        # Create a tqdm progress bar to display download progress
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

        # Open the file in binary write mode and write the content while updating the progress bar
        with open(save_path, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                progress_bar.update(len(data))

        # Close the progress bar
        progress_bar.close()

        print(f'Downloaded model weights to {save_path}')
    else:
        print(f'Failed to download model weights. HTTP status code: {response.status_code}')


if __name__ == '__main__':
    # IMAGENET1K_V1, for more information check torchvision.models.ViT_B_16_Weights
    ckpt_url = 'https://download.pytorch.org/models/vit_b_16-c867db91.pth'
    main(ckpt_url)

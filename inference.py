import requests
import base64
import urllib.parse
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import io
import os
import json
import argparse


def get_image_params(image_path):
    """
    Args:
        image_path (str): Path to the image you'd like to perform prediction on.

    Returns:
        Tuple containing a dict of requests kwargs.
    """
    hosted_image = urllib.parse.urlparse(image_path).scheme in ("http", "https")

    if hosted_image:
        return {"image": image_path}, {}

    image = Image.open(image_path)
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format="JPEG")
    data = MultipartEncoder(fields={"file": ("imageToUpload", buffered.getvalue(), "image/jpeg")})
    return (
        {},
        {"data": data, "headers": {"Content-Type": data.content_type}},
    )

def save_base64_image(base64_data, output_path):
    """Decode base64 data and save the image."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image_data = base64.b64decode(base64_data)
    with open(output_path, 'wb') as image_file:
        image_file.write(image_data)

def get_response(image_path, api_key):
    """Send the image to the inference API and return the response."""
    dataset_slug = "vms-all"  # Replace with your dataset's slug
    version = 4  # Replace with your model version
    url = f"https://detect.roboflow.com/{dataset_slug}/{version}"

    params, request_kwargs = get_image_params(image_path)

    params.update({
        "api_key": api_key,
        "confidence": 30,
        "overlap": 50,
        "format": "image_and_json",
        "stroke": 1,
        "labels": False
    })

    response = requests.post(
        url,
        params=params,
        **request_kwargs,
    )

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run inference on masked images and save outputs.")
    parser.add_argument("--masked_dir", type=str, required=True, help="Directory containing masked images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save inference images.")
    parser.add_argument("--json_dir", type=str, required=True, help="Directory to save inference JSON responses.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the inference API.")
    args = parser.parse_args()

    masked_dir = args.masked_dir
    output_dir = args.output_dir
    json_dir = args.json_dir
    api_key = args.api_key

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    for file_name in os.listdir(masked_dir):
        if file_name.endswith(".jpg"):
            image_path = os.path.join(masked_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            json_path = os.path.join(json_dir, f"{os.path.splitext(file_name)[0]}.json")

            print(f"Processing {file_name}...")

            # Run inference
            response = get_response(image_path, api_key)

            if response:
                # Save inference image
                save_base64_image(response['visualization'], output_path)

                # Save response JSON
                with open(json_path, 'w') as json_file:
                    json.dump(response, json_file)

                print(f"Processed {file_name}: Inference image and JSON saved.")
            else:
                print(f"Failed to process {file_name}.")

if __name__ == "__main__":
    main()

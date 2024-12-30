import requests
import base64
import urllib.parse
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import io
import os
import json

def get_image_params(image_path):
    """
    Get parameters about an image (i.e. dimensions) for use in an inference request.

    Args:
        image_path (str): path to the image you'd like to perform prediction on

    Returns:
        Tuple containing a dict of querystring params and a dict of requests kwargs

    Raises:
        Exception: Image path is not valid
    """

    hosted_image = urllib.parse.urlparse(image_path).scheme in ("http", "https")

    if hosted_image:
        image_dims = {"width": "Undefined", "height": "Undefined"}
        return {"image": image_path}, {}, image_dims

    image = Image.open(image_path)
    dimensions = image.size
    image_dims = {"width": str(dimensions[0]), "height": str(dimensions[1])}
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format="JPEG")
    data = MultipartEncoder(fields={"file": ("imageToUpload", buffered.getvalue(), "image/jpeg")})
    return (
        {},
        {"data": data, "headers": {"Content-Type": data.content_type}},
        image_dims,
    )


def save_base64_image(base64_data, output_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Decode the base64 data
    image_data = base64.b64decode(base64_data)

    # Write the decoded image data to a file
    with open(output_path, 'wb') as image_file:
        image_file.write(image_data)

def get_response(image_path):
    api_key = "M4k7un0DhO1ElBuNgY5i"
    dataset_slug = "vms-all"  # Replace with your dataset's slug
    version = 4  # Replace with your model version
    url = f"https://detect.roboflow.com/{dataset_slug}/{version}"

    params, request_kwargs, image_dims = get_image_params(image_path)

    params.update({
        "api_key": api_key,
        "confidence": 30,  # Prediction confidence threshold
        "overlap": 50,  # Bounding box overlap threshold
        "format": "image_and_json",  # The format you want (image and json in this case)
        "stroke": 1,  # Stroke width for bounding boxes
        "labels": False  # Include labels in the predictions
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

if __name__ == "__main__":
    raw_dir = "data/3796/masked"
    output_dir = "data/3796/output"
    json_dir = "data/3796/json"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    for file_name in os.listdir(raw_dir):
        if file_name.endswith(".jpg"):
            image_path = os.path.join(raw_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            json_path = os.path.join(json_dir, f"{os.path.splitext(file_name)[0]}.json")

            print(f"Processing {file_name}...")

            # Run inference
            response = get_response(image_path)

            if response:
                # Save inference image
                save_base64_image(response['visualization'], output_path)

                # Save response JSON
                with open(json_path, 'w') as json_file:
                    json.dump(response, json_file)

                print(f"Processed {file_name}: Inference image and JSON saved.")
            else:
                print(f"Failed to process {file_name}.")

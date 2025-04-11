import torch
from fastapi import FastAPI, File, UploadFile, Request
import io
from PIL import Image, ImageOps
import torchreid
import numpy as np
import torchvision.transforms as transforms
from fastapi.templating import Jinja2Templates
import os.path as osp
import json
import pickle
from functools import partial
from collections import OrderedDict
from torchreid.reid.data.transforms import build_transforms

app = FastAPI()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model_path = 'model/model_1744381991.pth'

json_path = model_path.replace('.pth', '.json')

# Load mapping information
with open(json_path, 'r') as json_file:
    model_data = json.load(json_file)

group_id_to_pid = model_data['group_id_to_pid']

model = torchreid.models.build_model(
    name="resnet50", 
    num_classes=42,
    pretrained=False,
    loss='softmax',
    use_gpu=True
)
# Load model state
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load the mappings
group_id_to_pid = model_data['group_id_to_pid']
transform_tr, transform_te = build_transforms(
    width=128,
    height=int(128 * 4 / 3)
)

def transform_image(img0):
    """Transforms a raw image (img0) k_tfm times with
    the transform function tfm.
    """
    img_list = []

    for k in range(1):
        img_list.append(transform_te(img0))

    img = img_list
    if len(img) == 1:
        img = img[0]

    return img.unsqueeze(0)

def resize_image(image):
    width = 128
    height = int(width * 4 / 3)  # Maintain 3:4 horizontal aspect ratio
    image = ImageOps.exif_transpose(image)  # Correct orientation based on EXIF data
    return image.resize((width, height))



from fastapi.staticfiles import StaticFiles
templates = Jinja2Templates(directory="templates")

# FastAPI endpoint to predict the label of an image
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Apply transformation to the image
    image = transform_image(image)
 
    with torch.no_grad():
        features = model(image)  # If this gives you 2048 features
        output = model.classifier(features)  # This should give 42 outputs
        _, predicted_pid = torch.max(output, 1)

    # Reverse the group_id_to_pid mapping to get the group_id (UUID)
    pid_to_group_id = {v: k for k, v in group_id_to_pid.items()}
    predicted_group_id = pid_to_group_id[predicted_pid.item()]

    return {
        "prediction": {
            "predicted_group_id": predicted_group_id,
        }
    }



@app.get("/")
async def get_image_upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



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

# To start: python -m uvicorn stick-it-ai-server:app --reload

app = FastAPI()

model_path = 'model/model_1744450299.pth'

json_path = model_path.replace('.pth', '.json')

# Load mapping information
with open(json_path, 'r') as json_file:
    model_data = json.load(json_file)

group_id_to_pid = model_data['group_id_to_pid']
pid_to_group_id = {v: k for k, v in model_data['group_id_to_pid'].items()}
pid_to_group_name = model_data['group_name_to_pid']

model = torchreid.models.build_model(
    name="resnet50", 
    num_classes=42,
    pretrained=False,
    loss='softmax',
    use_gpu=False
)
# Load model state
model.load_state_dict(torch.load(model_path))
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


from fastapi.staticfiles import StaticFiles
templates = Jinja2Templates(directory="templates")

# FastAPI endpoint to predict the label of an image
@app.post("/predict")
async def predict(file: UploadFile = File(...), group_ids: list[str] = None):
    # Read image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Apply transformation to the image
    image = transform_image(image)
    
    predicted_group_id = None
 
    with torch.no_grad():
        features = model(image)
        output = model.classifier(features)
        predicted_pid = torch.topk(output, k=output.size(1), dim=1).indices
        for pid in predicted_pid[0]:
            matching_group_id = pid_to_group_id.get(pid.item())
            if group_ids and matching_group_id in group_ids:
                predicted_group_id = matching_group_id
                break

   
    return {
        "predicted_group_id": predicted_group_id
    }



@app.get("/")
async def get_image_upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


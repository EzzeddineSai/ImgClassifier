import os
from fastapi import FastAPI, File, UploadFile
import io
from PIL import Image
import numpy as np
import mlflow

from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

S3_MODEL_URI = os.environ["S3_MODEL_URI"] 
pyfunc_model = mlflow.pyfunc.load_model(S3_MODEL_URI)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@asynccontextmanager # why is this needed?
async def lifespan(app: FastAPI):
    global pyfunc_model 
    print("Application is starting up...")

    yield # app runs here

    print("Application is shutting down...")    

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all origins
    allow_credentials=False,
    allow_methods=["*"],        # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],        # allow all headers
)

def transform_PIL_image(img): # make this async
  assert img.size == (32, 32), "Image size must be 32x32"
  img_data = np.array(img).astype(np.float32)  # shape (H, W, C)
  img_data = np.transpose(img_data, (2, 0, 1))   # change shape to (C, H, W) based on model expectation
  img_data /= 255.0 # normalize
  img_data = np.expand_dims(img_data, axis=0) # fit into a single batch tensor for inference
  return img_data

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)): # what is this strange type?????
    global pyfunc_model, CLASSES
    f = await file.read()
    img = Image.open(io.BytesIO(f))
   
    img_data = transform_PIL_image(img)
    model_output = pyfunc_model.predict(img_data)
    logits = model_output['logits'][0] # extract output from dict, and then take the first batch
    top_class = CLASSES[int(np.argmax(logits))]
    prediction = {"top_class": top_class, "logits":logits.tolist()}
    return prediction


@app.get("/") # get request and address
def root():
    return {"hello": "world"}
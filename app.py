import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import numpy as np
from mlflow import pyfunc

S3_MODEL_URI = os.environ["S3_MODEL_URI"] 
pyfunc_model = pyfunc.load_model(S3_MODEL_URI) # import model at startup

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def make_app():
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        # allow all origins
        allow_credentials=False,
        allow_methods=["*"],        # allow all HTTP methods (GET, POST, etc.)
        allow_headers=["*"],        # allow all headers
    )

    def transform_PIL_image(img):
        assert img.size == (32, 32), "Image size must be 32x32"
        img_data = np.array(img).astype(np.float32)  # shape (H, W, C)
        img_data = np.transpose(img_data, (2, 0, 1))   # change shape to (C, H, W) based on model expectation
        img_data /= 255.0 # normalize
        img_data = np.expand_dims(img_data, axis=0) # fit into a single batch tensor for inference
        return img_data

    @app.post("/uploadfile/")
    async def create_upload_file(file: UploadFile = File(...)):
        global pyfunc_model, CLASSES
        f = await file.read()
        img = Image.open(io.BytesIO(f))
        img_data = transform_PIL_image(img)
        model_output = pyfunc_model.predict(img_data)
        logits = model_output['logits'][0] # extract output from dict, and then take the first batch
        probs = np.exp(logits) / np.sum(np.exp(logits)) # softmax
        top_class = CLASSES[int(np.argmax(logits))]
        prediction = {"top_class": top_class, "probs":probs.tolist(), "class_names": CLASSES}
        return prediction

    @app.get("/") # get request and address
    def root():
        return {"message": "use the /uploadfile/ endpoint to POST an image for classification"}

    return app

app = make_app()

if __name__ == "__main__": # for development only
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True) # reload for dev only

else:
    from mangum import Mangum
    handler = Mangum(app) # for AWS Lambda deployment
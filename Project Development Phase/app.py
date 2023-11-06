from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from new_thing import load_model, img_to_array, load_img, Predict_cap, tokenizer
from tensorflow import keras
import numpy as np
from fastapi.responses import HTMLResponse
import tempfile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(os.getcwd(), "templates/static")), name="static")
# Enable CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to restrict origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("/home/kavin/API2/templates/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)



@app.get("/main.html", response_class=HTMLResponse)
async def get_main():
    with open("/home/kavin/API2/templates/main.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.post("/upload/")
async def upload_file(img: UploadFile):
    # Check if the uploaded file is an image
    if not img.content_type.startswith('image/'):
        return JSONResponse(content={"error": "Uploaded file is not an image"}, status_code=400)

    try:
        # Read the image file into memory
        contents = await img.read()
        image = Image.open(io.BytesIO(contents))

        # Generate a caption for the uploaded image
        extraction = load_model("extraction_model.h5")

        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
            image.save(tmpfile.name, format="JPEG")

        img_path = tmpfile.name  # Get the path to the temporary file
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        feature = extraction.predict(img, verbose=0)
        model = keras.models.load_model("overall_model.h5")
        caption = Predict_cap(model, feature, tokenizer, 74)

        return {"Caption": caption}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import soundfile as sf
import numpy as np
import torch
import os

from Voice_authentication.feature_extraction import extract_features
from Voice_authentication.model import get_speaker_encoder

app = FastAPI(title="Voice Biometric API")

# Allow local browser frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend directory when available
HERE = os.path.dirname(__file__)
FRONTEND_DIR = os.path.abspath(os.path.join(HERE, '..', 'frontend'))
if os.path.isdir(FRONTEND_DIR):
    app.mount('/ui', StaticFiles(directory=FRONTEND_DIR, html=True), name='frontend')

# Load encoder once on startup
MODEL_PATH = os.environ.get("VB_MODEL", "models/CNN.pt")
DEVICE = torch.device("cpu")
encoder = None


@app.on_event("startup")
def load_model():
    global encoder
    encoder = get_speaker_encoder(MODEL_PATH)


@app.post("/api/embed")
async def embed(audio_file: UploadFile = File(...)):
    """Accepts an audio file (wav/flac) and returns a 1-D embedding vector."""
    data = await audio_file.read()
    try:
        arr, sr = sf.read(io.BytesIO(data))
    except Exception as e:
        return JSONResponse({"error": f"cannot read audio: {e}"}, status_code=400)

    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)

    feats = extract_features(arr, sr)
    with torch.no_grad():
        inp = torch.from_numpy(feats).unsqueeze(0).float()
        out = encoder(inp.to(DEVICE))
    vec = out.squeeze(0).cpu().numpy().tolist()
    return {"embedding": vec}


@app.post("/api/compare")
async def compare(audio_a: UploadFile = File(...), audio_b: UploadFile = File(...)):
    """Accepts two audio files and returns both embeddings and cosine similarity."""
    data_a = await audio_a.read()
    data_b = await audio_b.read()
    try:
        arr_a, sr_a = sf.read(io.BytesIO(data_a))
        arr_b, sr_b = sf.read(io.BytesIO(data_b))
    except Exception as e:
        return JSONResponse({"error": f"cannot read audio: {e}"}, status_code=400)

    if hasattr(arr_a, 'ndim') and arr_a.ndim > 1:
        arr_a = np.mean(arr_a, axis=1)
    if hasattr(arr_b, 'ndim') and arr_b.ndim > 1:
        arr_b = np.mean(arr_b, axis=1)

    feats_a = extract_features(arr_a, sr_a)
    feats_b = extract_features(arr_b, sr_b)

    with torch.no_grad():
        inp_a = torch.from_numpy(feats_a).unsqueeze(0).float().to(DEVICE)
        inp_b = torch.from_numpy(feats_b).unsqueeze(0).float().to(DEVICE)
        out_a = encoder(inp_a)
        out_b = encoder(inp_b)

    vec_a = out_a.squeeze(0).cpu().numpy()
    vec_b = out_b.squeeze(0).cpu().numpy()

    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    cos_sim = float(0.0 if denom == 0 else float(np.dot(vec_a, vec_b) / denom))

    return {
        "embedding_a": vec_a.tolist(),
        "embedding_b": vec_b.tolist(),
        "cosine": cos_sim,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

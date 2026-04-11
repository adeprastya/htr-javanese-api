import io
import logging
from contextlib import asynccontextmanager
from functools import lru_cache

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from src.cnn_bilstm import CNNBiLSTM
from src.ctc_decoder import best_path_decode
from src.preprocessing import get_preprocessing_pipeline
from src.vocabulary import BLANK_IDX, IDX2CHAR, NUM_CLASSES

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_path: str = "model/5cnn-2bilstm.pth"
    img_height: int = 48
    cnn_layers: int = 5
    rnn_layers: int = 2
    max_image_bytes: int = 4 * 1024 * 1024  # 4 MB

    class Config:
        env_prefix = "APP_"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model state (populated at startup)
# ---------------------------------------------------------------------------

_state: dict = {}


def load_model(settings: Settings) -> tuple:
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = CNNBiLSTM(
        num_classes=NUM_CLASSES,
        cnn_layers=settings.cnn_layers,
        rnn_layers=settings.rnn_layers,
    )
    checkpoint = torch.load(settings.model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s", settings.model_path)

    preprocess = get_preprocessing_pipeline(img_height=settings.img_height)
    return model, preprocess, device


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _state["model"], _state["preprocess"], _state["device"] = load_model(settings)
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    status: str
    prediction: str


class ErrorResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="API Pengenalan Teks Aksara Jawa Nglegena",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def validate_and_open_image(contents: bytes) -> Image.Image:
    """Validate raw bytes and return an RGB PIL image."""
    settings = get_settings()
    if len(contents) > settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size must be ≤ {settings.max_image_bytes / 1024 / 1024:.0f} MB",
        )
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupted image file",
        ) from exc


def run_inference(img: Image.Image) -> str:
    """Run the full preprocessing → model → decode pipeline."""
    model = _state["model"]
    preprocess = _state["preprocess"]
    device = _state["device"]

    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)

    logits = output.argmax(dim=2)[0].tolist()
    return best_path_decode(logits, IDX2CHAR, BLANK_IDX)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "status": "success",
        "message": "Welcome to Pengenalan Teks Aksara Jawa Nglegena API",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    await image.close()

    img = validate_and_open_image(contents)

    try:
        prediction = run_inference(img)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc

    return PredictionResponse(status="success", prediction=prediction)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(status="error", message=exc.detail).model_dump(),
    )

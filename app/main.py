"""FastAPI application entry point."""

import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from app.routers import transcribe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="AutoGLM ASR Web Client")

# Include routers
app.include_router(transcribe.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Redirect root to index.html."""
    return RedirectResponse(url="/static/index.html")


# Mount static files - must be last to not override API routes
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

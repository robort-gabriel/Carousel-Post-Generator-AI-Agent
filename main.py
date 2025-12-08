"""
FastAPI Application for Carousel Post Generator

Production-ready FastAPI application with security features:
- API key authentication
- Rate limiting
- Input validation
- CORS configuration
- Comprehensive error handling
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Security, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator, HttpUrl
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from carousel_post_generator_agent import create_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Rate limiter configuration
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
limiter = Limiter(key_func=get_remote_address)

# API Key Security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Get API key from environment
API_KEY = os.getenv("API_KEY", "")
if not API_KEY:
    logger.warning(
        "API_KEY not set in environment variables. API will be accessible without authentication."
    )


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """
    Verify API key from request header.

    Args:
        api_key: API key from request header

    Returns:
        True if API key is valid

    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not API_KEY:
        # If no API key is configured, allow all requests
        return True

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing. Please provide X-API-Key header.",
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return True


# Request/Response Models
class CarouselRequest(BaseModel):
    """Request model for carousel post generation."""

    url: HttpUrl = Field(
        ...,
        description="Article URL to convert into carousel post",
    )
    max_slides: int = Field(
        10,
        ge=3,
        le=15,
        description="Maximum number of slides to generate (3-15, default: 10)",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: HttpUrl) -> str:
        """Convert HttpUrl to string."""
        return str(v)


class SlideImage(BaseModel):
    """Slide image model."""

    slide_number: int
    image_path: Optional[str] = None
    image_filename: Optional[str] = None
    image_relative_path: Optional[str] = None
    image_prompt: Optional[str] = None


class CarouselSlide(BaseModel):
    """Carousel slide model."""

    slide_number: int
    title: str
    content: str
    image_prompt: Optional[str] = None
    image_path: Optional[str] = None
    image_filename: Optional[str] = None
    image_relative_path: Optional[str] = None


class CarouselResponse(BaseModel):
    """Response model for carousel generation results."""

    status: str
    url: str
    article_title: str
    total_slides: int
    slides: List[CarouselSlide]
    processing_status: str


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    message: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    error_type: Optional[str] = None


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the FastAPI app."""
    logger.info("Starting Carousel Post Generator API...")
    yield
    logger.info("Shutting down Carousel Post Generator API...")


# Create FastAPI app
app = FastAPI(
    title="Carousel Post Generator API",
    description="Production-ready API for generating carousel posts from article URLs with AI-generated images",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
)
async def health_check():
    """
    Health check endpoint to verify API is running.

    Returns:
        Health status and API version
    """
    return HealthResponse(
        status="healthy",
        message="Carousel Post Generator API is running",
        version="1.0.0",
    )


@app.post(
    "/api/v1/generate",
    response_model=CarouselResponse,
    status_code=status.HTTP_200_OK,
    tags=["Carousel Generation"],
    summary="Generate carousel post from article URL",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(
    f"{RATE_LIMIT_PER_MINUTE}/minute"
)  # Rate limit from environment variable
async def generate_carousel(
    request_data: CarouselRequest,
    request: Request,
):
    """
    Generate a carousel post from an article URL.

    This endpoint:
    - Scrapes the article content from the provided URL
    - Uses AI to create engaging carousel slide content
    - Generates custom images for each slide using Gemini via OpenRouter
    - Saves images locally in the output folder
    - Returns carousel slides with image paths

    Args:
        request_data: Carousel request with URL and max_slides
        request: FastAPI Request object for rate limiting

    Returns:
        CarouselResponse with generated slides and images

    Raises:
        HTTPException: If generation fails or request is invalid
    """
    try:
        logger.info(
            f"Received carousel generation request: url='{request_data.url}', "
            f"max_slides={request_data.max_slides}"
        )

        # Create agent instance
        agent = create_agent()

        # Process the carousel generation request
        result = await agent.process(
            url=request_data.url,
            max_slides=request_data.max_slides,
        )

        # Convert slides to response model
        carousel_slides = []
        for slide in result.get("slides", []):
            carousel_slide = CarouselSlide(
                slide_number=slide.get("slide_number", 0),
                title=slide.get("title", ""),
                content=slide.get("content", ""),
                image_prompt=slide.get("image_prompt"),
                image_path=slide.get("image_path"),
                image_filename=slide.get("image_filename"),
                image_relative_path=slide.get("image_relative_path"),
            )
            carousel_slides.append(carousel_slide)

        logger.info(
            f"Successfully generated {result.get('total_slides', 0)} slides for URL: {request_data.url}"
        )

        return CarouselResponse(
            status=result.get("status", "success"),
            url=result.get("url", request_data.url),
            article_title=result.get("article_title", "Unknown"),
            total_slides=result.get("total_slides", 0),
            slides=carousel_slides,
            processing_status=result.get("processing_status", "completed"),
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error generating carousel post: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Carousel generation failed: {str(e)}",
        )


@app.get(
    "/",
    tags=["Root"],
    summary="API root endpoint",
)
async def root():
    """
    Root endpoint with API information.

    Returns:
        API information and available endpoints
    """
    return {
        "name": "Carousel Post Generator API",
        "version": "1.0.0",
        "description": "Production-ready API for generating carousel posts from article URLs with AI-generated images",
        "endpoints": {
            "health": "/health",
            "generate": "/api/v1/generate",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "False").lower() == "true",
        log_level="info",
    )

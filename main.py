"""
FastAPI Application for Social Media Content Generator

Production-ready FastAPI application with security features:
- API key authentication
- Rate limiting
- Input validation
- CORS configuration
- Comprehensive error handling
"""

import logging
import os
import uuid
import time
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Security,
    status,
    Request,
    UploadFile,
    File,
    Form,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, HttpUrl, ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from carousel_post_generator_agent import (
    create_agent,
    generate_single_informational_image,
    generate_infographic_from_prompt,
    generate_infographic_with_reference_image,
    generate_carousel_from_prompt,
)

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
        description="Article URL to convert into social media carousel post",
    )
    max_slides: int = Field(
        10,
        ge=3,
        le=15,
        description="Maximum number of slides to generate (3-15, default: 10)",
    )
    username: Optional[str] = Field(
        None,
        description="Social media username to display on slides (e.g., '@robots'). Defaults to '@robots' if not provided.",
    )
    tagline: Optional[str] = Field(
        None,
        description="Tagline/brand message to display on slides (e.g., 'daily programming tips & tricks'). Defaults to 'daily programming tips & tricks' if not provided.",
    )
    title: Optional[str] = Field(
        None,
        description="Custom title to override the scraped article title. If provided, this title will be used instead of the article's original title.",
    )
    extra_instructions: Optional[str] = Field(
        None,
        description="Additional instructions for the LLM to customize the carousel generation (e.g., 'Use a minimalist design style' or 'Focus on beginner-friendly language').",
    )
    font_name: Optional[str] = Field(
        None,
        description="Font name to use for the carousel slides (e.g., 'Arial', 'Roboto', 'Montserrat'). Defaults to a modern sans-serif font if not provided.",
    )
    background_info: Optional[str] = Field(
        None,
        description="Background description for the carousel slides (e.g., 'dark navy gradient', 'light mode with subtle tech theme', 'purple to blue gradient'). Defaults to 'Clean gradient or subtle tech/coding theme (dark navy/blue/purple or modern light mode)' if not provided.",
    )
    color_schema: Optional[str] = Field(
        None,
        description="Color schema description for the carousel slides (e.g., 'navy background with white and cyan accent text', 'light background with dark blue text', 'purple gradient with white text'). Defaults to 'Consistent across all slides (e.g., navy background, white + cyan accent text)' if not provided.",
    )
    image_provider: Optional[str] = Field(
        None,
        description="Image generation provider ('openrouter' or 'openai'). Defaults to 'openrouter' if not provided. Use 'openai' for DALL-E models.",
    )
    image_model: Optional[str] = Field(
        None,
        description="Image generation model name. For OpenAI: 'dall-e-3' or 'dall-e-2'. For OpenRouter: 'google/gemini-2.5-flash-image', 'google/gemini-3-pro-image-preview', etc. Defaults to provider's default model if not provided.",
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
    image_url: Optional[str] = None  # URL to access the image via API
    image_filename: Optional[str] = None
    image_relative_path: Optional[str] = None


class CarouselResponse(BaseModel):
    """Response model for carousel generation results."""

    success: bool = True
    status: str
    url: str
    article_title: str
    total_slides: int
    slides: List[CarouselSlide]
    processing_status: str
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    message: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = False
    error: str
    detail: Optional[str] = None
    error_type: Optional[str] = None
    request_id: Optional[str] = None


class SuccessResponse(BaseModel):
    """Base success response model."""

    success: bool = True
    message: Optional[str] = None
    request_id: Optional[str] = None


class SingleImageRequest(BaseModel):
    """Request model for single informational image generation."""

    url: HttpUrl = Field(
        ...,
        description="Article URL to convert into single informational image",
    )
    username: Optional[str] = Field(
        None,
        description="Social media username to display on image (e.g., '@robots'). Defaults to '@coding_robort' if not provided.",
    )
    tagline: Optional[str] = Field(
        None,
        description="Tagline/brand message to display on image (e.g., 'daily programming tips & tricks'). Defaults to 'daily programming tips & tricks' if not provided.",
    )
    title: Optional[str] = Field(
        None,
        description="Custom title to override the scraped article title. If provided, this title will be used instead of the article's original title.",
    )
    extra_instructions: Optional[str] = Field(
        None,
        description="Additional instructions for the LLM to customize the image generation (e.g., 'Use a minimalist design style' or 'Focus on beginner-friendly language').",
    )
    font_name: Optional[str] = Field(
        None,
        description="Font name to use for the informational image (e.g., 'Arial', 'Roboto', 'Montserrat'). Defaults to a modern sans-serif font if not provided.",
    )
    background_info: Optional[str] = Field(
        None,
        description="Background description for the informational image (e.g., 'dark navy gradient', 'light mode with subtle tech theme', 'purple to blue gradient'). Defaults to 'Clean gradient or subtle tech/coding theme (dark navy/blue/purple or modern light mode)' if not provided.",
    )
    color_schema: Optional[str] = Field(
        None,
        description="Color schema description for the informational image (e.g., 'navy background with white and cyan accent text', 'light background with dark blue text', 'purple gradient with white text'). Defaults to 'Modern and visually appealing (e.g., navy background, white + cyan accent text)' if not provided.",
    )
    image_provider: Optional[str] = Field(
        None,
        description="Image generation provider ('openrouter' or 'openai'). Defaults to 'openrouter' if not provided. Use 'openai' for DALL-E models.",
    )
    image_model: Optional[str] = Field(
        None,
        description="Image generation model name. For OpenAI: 'dall-e-3' or 'dall-e-2'. For OpenRouter: 'google/gemini-2.5-flash-image', 'google/gemini-3-pro-image-preview', etc. Defaults to provider's default model if not provided.",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: HttpUrl) -> str:
        """Convert HttpUrl to string."""
        return str(v)


class SingleImageResponse(BaseModel):
    """Response model for single informational image generation."""

    success: bool = True
    status: str
    url: str
    article_title: str
    title: str
    image_path: Optional[str] = None
    image_url: Optional[str] = None  # URL to access the image via API
    request_id: Optional[str] = None


class InfographicFromPromptRequest(BaseModel):
    """Request model for infographic generation from text prompt."""

    prompt: str = Field(
        ...,
        description="Text prompt describing what infographic to create (e.g., 'top 10 free AI tools', 'how to use React hooks', 'benefits of TypeScript')",
    )
    username: Optional[str] = Field(
        None,
        description="Social media username to display on image (e.g., '@robots'). Defaults to '@coding_robort' if not provided.",
    )
    tagline: Optional[str] = Field(
        None,
        description="Tagline/brand message to display on image (e.g., 'daily programming tips & tricks'). Defaults to 'daily programming tips & tricks' if not provided.",
    )
    font_name: Optional[str] = Field(
        None,
        description="Font name to use for the infographic (e.g., 'Arial', 'Roboto', 'Montserrat'). Defaults to a modern sans-serif font if not provided.",
    )
    background_info: Optional[str] = Field(
        None,
        description="Background description for the infographic (e.g., 'dark navy gradient', 'light mode with subtle tech theme', 'purple to blue gradient'). Defaults to 'Clean gradient or subtle tech/coding theme (dark navy/blue/purple or modern light mode)' if not provided.",
    )
    color_schema: Optional[str] = Field(
        None,
        description="Color schema description for the infographic (e.g., 'navy background with white and cyan accent text', 'light background with dark blue text', 'purple gradient with white text'). Defaults to 'Modern and visually appealing (e.g., navy background, white + cyan accent text)' if not provided.",
    )
    extra_instructions: Optional[str] = Field(
        None,
        description="Additional instructions for the LLM to customize the image generation (e.g., 'Use a minimalist design style' or 'Focus on beginner-friendly language').",
    )
    image_provider: Optional[str] = Field(
        None,
        description="Image generation provider ('openrouter' or 'openai'). Defaults to 'openrouter' if not provided. Use 'openai' for DALL-E models.",
    )
    image_model: Optional[str] = Field(
        None,
        description="Image generation model name. For OpenAI: 'dall-e-3' or 'dall-e-2'. For OpenRouter: 'google/gemini-2.5-flash-image', 'google/gemini-3-pro-image-preview', etc. Defaults to provider's default model if not provided.",
    )


class CarouselFromPromptRequest(BaseModel):
    """Request model for carousel generation from text prompt."""

    prompt: str = Field(
        ...,
        description="Text prompt describing what carousel to create (e.g., 'top 10 free AI tools', 'how to learn React', 'benefits of TypeScript')",
    )
    max_slides: int = Field(
        10,
        ge=3,
        le=15,
        description="Maximum number of slides to generate (3-15, default: 10)",
    )
    username: Optional[str] = Field(
        None,
        description="Social media username to display on slides (e.g., '@robots'). Defaults to '@coding_robort' if not provided.",
    )
    tagline: Optional[str] = Field(
        None,
        description="Tagline/brand message to display on slides (e.g., 'daily programming tips & tricks'). Defaults to 'daily programming tips & tricks' if not provided.",
    )
    font_name: Optional[str] = Field(
        None,
        description="Font name to use for the carousel slides (e.g., 'Arial', 'Roboto', 'Montserrat'). Defaults to a modern sans-serif font if not provided.",
    )
    background_info: Optional[str] = Field(
        None,
        description="Background description for the carousel slides (e.g., 'dark navy gradient', 'light mode with subtle tech theme', 'purple to blue gradient'). Defaults to 'Clean gradient or subtle tech/coding theme (dark navy/blue/purple or modern light mode)' if not provided.",
    )
    color_schema: Optional[str] = Field(
        None,
        description="Color schema description for the carousel slides (e.g., 'navy background with white and cyan accent text', 'light background with dark blue text', 'purple gradient with white text'). Defaults to 'Consistent across all slides (e.g., navy background, white + cyan accent text)' if not provided.",
    )
    extra_instructions: Optional[str] = Field(
        None,
        description="Additional instructions for the LLM to customize the carousel generation (e.g., 'Use a minimalist design style' or 'Focus on beginner-friendly language').",
    )
    image_provider: Optional[str] = Field(
        None,
        description="Image generation provider ('openrouter' or 'openai'). Defaults to 'openrouter' if not provided. Use 'openai' for DALL-E models.",
    )
    image_model: Optional[str] = Field(
        None,
        description="Image generation model name. For OpenAI: 'dall-e-3' or 'dall-e-2'. For OpenRouter: 'google/gemini-2.5-flash-image', 'google/gemini-3-pro-image-preview', etc. Defaults to provider's default model if not provided.",
    )


class InfographicFromPromptResponse(BaseModel):
    """Response model for infographic generation from prompt."""

    success: bool = True
    status: str
    prompt: str
    type: str
    title: str
    content: str
    image_path: Optional[str] = None
    image_url: Optional[str] = None  # URL to access the image via API
    image_filename: Optional[str] = None
    image_relative_path: Optional[str] = None
    request_id: Optional[str] = None


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the FastAPI app."""
    logger.info("Starting Social Media Content Generator API...")
    yield
    logger.info("Shutting down Social Media Content Generator API...")


# Create FastAPI app
app = FastAPI(
    title="Social Media Content Generator API",
    description="Production-ready API for generating social media content (carousel posts and single informational images) from article URLs with AI-generated images",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration - Frontend-friendly
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if "*" not in cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)


# Request ID and Response Time Middleware
@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    """Add request ID and track response time for all requests."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()

    response = await call_next(request)

    # Add headers
    response.headers["X-Request-ID"] = request_id
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 3))

    return response


# Global Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with consistent JSON format."""
    request_id = getattr(request.state, "request_id", None)
    errors = exc.errors()
    error_details = []
    for error in errors:
        error_details.append(
            {
                "field": ".".join(str(x) for x in error.get("loc", [])),
                "message": error.get("msg"),
                "type": error.get("type"),
            }
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation Error",
            "detail": "Invalid request data",
            "errors": error_details,
            "error_type": "validation_error",
            "request_id": request_id,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent JSON format."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "detail": exc.detail,
            "error_type": "http_exception",
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions with consistent JSON format."""
    request_id = getattr(request.state, "request_id", None)
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "error_type": "internal_error",
            "request_id": request_id,
        },
    )


# Helper Functions
def get_image_url(image_path: Optional[str], request: Request) -> Optional[str]:
    """
    Generate a URL to access an image via the API.

    Args:
        image_path: Local file path to the image
        request: FastAPI Request object

    Returns:
        URL to access the image, or None if image_path is None
    """
    if not image_path:
        return None

    try:
        # Extract filename from path
        filename = Path(image_path).name
        # Generate URL using the image serving endpoint
        base_url = str(request.base_url).rstrip("/")
        return f"{base_url}/api/v1/images/{filename}"
    except Exception:
        return None


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
        message="Social Media Content Generator API is running",
        version="1.0.0",
    )


@app.get(
    "/api/v1/images/{filename}",
    tags=["Images"],
    summary="Serve generated images",
    response_class=FileResponse,
)
async def serve_image(filename: str):
    """
    Serve generated images from the output directory.

    Args:
        filename: Name of the image file to serve

    Returns:
        Image file or 404 if not found
    """
    try:
        # Get output directory
        try:
            script_dir = Path(__file__).parent.absolute()
            base_output_dir = script_dir / "output"
        except NameError:
            base_output_dir = Path.cwd() / "output"

        # Search for the image in all subdirectories
        image_path = None
        for subdir in base_output_dir.rglob(filename):
            if subdir.is_file():
                image_path = subdir
                break

        if not image_path or not image_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image '{filename}' not found",
            )

        return FileResponse(
            path=str(image_path),
            media_type="image/png",
            filename=filename,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error serving image: {str(e)}",
        )


@app.post(
    "/api/v1/generate",
    response_model=CarouselResponse,
    status_code=status.HTTP_200_OK,
    tags=["Social Media Content"],
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
            username=request_data.username,
            tagline=request_data.tagline,
            title=request_data.title,
            extra_instructions=request_data.extra_instructions,
            font_name=request_data.font_name,
            background_info=request_data.background_info,
            color_schema=request_data.color_schema,
            image_provider=request_data.image_provider,
            image_model=request_data.image_model,
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

        # Add image URLs to slides
        for slide in carousel_slides:
            if slide.image_filename:
                slide.image_url = get_image_url(
                    slide.image_path, request
                )  # type: ignore

        request_id = getattr(request.state, "request_id", None)
        return CarouselResponse(
            success=True,
            status=result.get("status", "success"),
            url=result.get("url", request_data.url),
            article_title=result.get("article_title", "Unknown"),
            total_slides=result.get("total_slides", 0),
            slides=carousel_slides,
            processing_status=result.get("processing_status", "completed"),
            request_id=request_id,
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        request_id = getattr(request.state, "request_id", None)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error generating social media content: {str(e)}", exc_info=True)
        request_id = getattr(request.state, "request_id", None)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Carousel generation failed: {str(e)}",
        )


@app.post(
    "/api/v1/generate-single-image",
    response_model=SingleImageResponse,
    status_code=status.HTTP_200_OK,
    tags=["Social Media Content"],
    summary="Generate single informational image from article URL",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(
    f"{RATE_LIMIT_PER_MINUTE}/minute"
)  # Rate limit from environment variable
async def generate_single_image(
    request_data: SingleImageRequest,
    request: Request,
):
    """
    Generate a single informational image from an article URL.

    This endpoint:
    - Scrapes the article content from the provided URL
    - Analyzes if the article is a list post or general article
    - If list post: Extracts list items and creates one image with the list
    - If general article: Summarizes and creates one image with key takeaways
    - Generates custom image using Gemini via OpenRouter
    - Saves image locally in the output folder
    - Returns image information and metadata

    Args:
        request_data: Single image request with URL and optional parameters
        request: FastAPI Request object for rate limiting

    Returns:
        SingleImageResponse with generated image information

    Raises:
        HTTPException: If generation fails or request is invalid
    """
    try:
        logger.info(
            f"Received single image generation request: url='{request_data.url}'"
        )

        # Generate the single informational image
        result = await generate_single_informational_image(
            url=request_data.url,
            username=request_data.username,
            tagline=request_data.tagline,
            title=request_data.title,
            extra_instructions=request_data.extra_instructions,
            font_name=request_data.font_name,
            background_info=request_data.background_info,
            color_schema=request_data.color_schema,
            image_provider=request_data.image_provider,
            image_model=request_data.image_model,
        )

        logger.info(f"Successfully generated single image for URL: {request_data.url}")

        request_id = getattr(request.state, "request_id", None)
        return SingleImageResponse(
            success=True,
            status=result.get("status", "success"),
            url=result.get("url", request_data.url),
            article_title=result.get("article_title", "Unknown"),
            title=result.get("title", ""),
            image_path=result.get("image_path"),
            image_url=get_image_url(result.get("image_path"), request),
            request_id=request_id,
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error generating single image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image generation failed: {str(e)}",
        )


@app.post(
    "/api/v1/generate-infographic-from-prompt",
    response_model=InfographicFromPromptResponse,
    status_code=status.HTTP_200_OK,
    tags=["Social Media Content"],
    summary="Generate infographic from text prompt",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(
    f"{RATE_LIMIT_PER_MINUTE}/minute"
)  # Rate limit from environment variable
async def generate_infographic_from_prompt_endpoint(
    request_data: InfographicFromPromptRequest,
    request: Request,
):
    """
    Generate an infographic from a user text prompt.

    This endpoint:
    - Takes a text prompt from the user (e.g., "top 10 free AI tools")
    - Analyzes the prompt to determine if it's a list type or summary type
    - If list type: Generates the list items based on the prompt (e.g., creates list of 10 AI tools)
    - If summary type: Creates a summary infographic with key points
    - Generates custom image using Gemini via OpenRouter
    - Saves image locally in the output folder
    - Returns infographic information and metadata

    Examples of prompts:
    - List type: "top 10 free AI tools", "5 best React libraries", "7 ways to improve productivity"
    - Summary type: "how to use React hooks", "explain machine learning", "benefits of TypeScript"

    Args:
        request_data: Infographic request with prompt and optional design parameters
        request: FastAPI Request object for rate limiting

    Returns:
        InfographicFromPromptResponse with generated infographic information

    Raises:
        HTTPException: If generation fails or request is invalid
    """
    try:
        logger.info(
            f"Received infographic generation request: prompt='{request_data.prompt}'"
        )

        # Generate the infographic from prompt
        result = await generate_infographic_from_prompt(
            user_prompt=request_data.prompt,
            username=request_data.username,
            tagline=request_data.tagline,
            font_name=request_data.font_name,
            background_info=request_data.background_info,
            color_schema=request_data.color_schema,
            extra_instructions=request_data.extra_instructions,
            image_provider=request_data.image_provider,
            image_model=request_data.image_model,
        )

        logger.info(
            f"Successfully generated infographic from prompt: {request_data.prompt}"
        )

        request_id = getattr(request.state, "request_id", None)
        return InfographicFromPromptResponse(
            success=True,
            status=result.get("status", "success"),
            prompt=result.get("prompt", request_data.prompt),
            type=result.get("type", "unknown"),
            title=result.get("title", ""),
            content=result.get("content", ""),
            image_path=result.get("image_path"),
            image_url=get_image_url(result.get("image_path"), request),
            image_filename=result.get("image_filename"),
            image_relative_path=result.get("image_relative_path"),
            request_id=request_id,
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}",
        )
    except Exception as e:
        logger.error(
            f"Error generating infographic from prompt: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Infographic generation failed: {str(e)}",
        )


@app.post(
    "/api/v1/generate-carousel-from-prompt",
    response_model=CarouselResponse,
    status_code=status.HTTP_200_OK,
    tags=["Social Media Content"],
    summary="Generate carousel post from text prompt",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(
    f"{RATE_LIMIT_PER_MINUTE}/minute"
)  # Rate limit from environment variable
async def generate_carousel_from_prompt_endpoint(
    request_data: CarouselFromPromptRequest,
    request: Request,
):
    """
    Generate a carousel post from a user text prompt.

    This endpoint:
    - Takes a text prompt from the user (e.g., "top 10 free AI tools", "how to learn React")
    - Analyzes the prompt to understand what content to create (list type or guide type)
    - Generates carousel slides with content based on the prompt
    - Generates custom images for each slide using the configured image provider
    - Saves images locally in the output folder
    - Returns carousel slides with image paths and URLs

    Examples of prompts:
    - List type: "top 10 free AI tools", "5 best React libraries", "7 ways to improve productivity"
    - Guide type: "how to learn React", "explain machine learning", "benefits of TypeScript"

    Args:
        request_data: Carousel from prompt request with prompt and optional parameters
        request: FastAPI Request object for rate limiting

    Returns:
        CarouselResponse with generated slides and images

    Raises:
        HTTPException: If generation fails or request is invalid
    """
    try:
        logger.info(
            f"Received carousel from prompt generation request: prompt='{request_data.prompt}', "
            f"max_slides={request_data.max_slides}"
        )

        # Generate the carousel from prompt
        result = await generate_carousel_from_prompt(
            user_prompt=request_data.prompt,
            max_slides=request_data.max_slides,
            username=request_data.username,
            tagline=request_data.tagline,
            font_name=request_data.font_name,
            background_info=request_data.background_info,
            color_schema=request_data.color_schema,
            extra_instructions=request_data.extra_instructions,
            image_provider=request_data.image_provider,
            image_model=request_data.image_model,
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
            f"Successfully generated {result.get('total_slides', 0)} slides from prompt: {request_data.prompt}"
        )

        # Add image URLs to slides
        for slide in carousel_slides:
            if slide.image_filename:
                slide.image_url = get_image_url(
                    slide.image_path, request
                )  # type: ignore

        request_id = getattr(request.state, "request_id", None)
        return CarouselResponse(
            success=True,
            status=result.get("status", "success"),
            url="",  # No URL for prompt-based generation
            article_title=result.get("article_title", "Unknown"),
            total_slides=result.get("total_slides", 0),
            slides=carousel_slides,
            processing_status=result.get("processing_status", "completed"),
            request_id=request_id,
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error generating carousel from prompt: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Carousel generation failed: {str(e)}",
        )


@app.post(
    "/api/v1/generate-infographic-with-reference",
    response_model=InfographicFromPromptResponse,
    status_code=status.HTTP_200_OK,
    tags=["Social Media Content"],
    summary="Generate infographic from prompt using reference image design",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(
    f"{RATE_LIMIT_PER_MINUTE}/minute"
)  # Rate limit from environment variable
async def generate_infographic_with_reference_endpoint(
    request: Request,
    reference_image: UploadFile = File(
        ..., description="Reference image file to use as design guide"
    ),
    prompt: str = Form(
        ..., description="Text prompt describing what infographic to create"
    ),
    username: Optional[str] = Form(
        None, description="Social media username (e.g., '@robots')"
    ),
    tagline: Optional[str] = Form(None, description="Tagline/brand message"),
    font_name: Optional[str] = Form(None, description="Font name for the infographic"),
    background_info: Optional[str] = Form(None, description="Background description"),
    color_schema: Optional[str] = Form(None, description="Color schema description"),
    image_provider: Optional[str] = Form(
        None, description="Image generation provider ('openrouter' or 'openai')"
    ),
    image_model: Optional[str] = Form(
        None,
        description="Image generation model (e.g., 'dall-e-3', 'google/gemini-2.5-flash-image')",
    ),
):
    """
    Generate an infographic from a text prompt using a reference image as design guide.

    This endpoint:
    - Accepts an uploaded reference image file
    - Takes a text prompt describing what infographic to create
    - Uses the reference image as a design style guide
    - Analyzes the prompt to determine if it's a list type or summary type
    - Generates the infographic content matching the reference image's design style
    - Creates the new infographic using Gemini via OpenRouter
    - Saves image locally in the output folder
    - Returns infographic information and metadata

    The reference image will be used to match:
    - Design style and aesthetic
    - Color scheme
    - Layout structure
    - Typography
    - Visual elements

    Examples of prompts:
    - List type: "top 10 free AI tools", "5 best React libraries", "7 ways to improve productivity"
    - Summary type: "how to use React hooks", "explain machine learning", "benefits of TypeScript"

    Args:
        request: FastAPI Request object for rate limiting
        reference_image: Uploaded image file to use as design reference
        prompt: Text prompt describing what infographic to create
        username: Optional social media username
        tagline: Optional tagline/brand message
        font_name: Optional font name
        background_info: Optional background description
        color_schema: Optional color schema description

    Returns:
        InfographicFromPromptResponse with generated infographic information

    Raises:
        HTTPException: If generation fails or request is invalid
    """
    try:
        # Validate image file
        if (
            not reference_image.content_type
            or not reference_image.content_type.startswith("image/")
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Please upload an image file (PNG, JPG, JPEG, etc.)",
            )

        # Read image file
        image_bytes = await reference_image.read()

        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file. Please upload a valid image.",
            )

        # Validate file size (max 10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image file too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024)}MB",
            )

        logger.info(
            f"Received infographic generation request with reference image: prompt='{prompt}', "
            f"image_size={len(image_bytes)} bytes, content_type={reference_image.content_type}"
        )

        # Generate the infographic with reference image
        result = await generate_infographic_with_reference_image(
            user_prompt=prompt,
            reference_image_bytes=image_bytes,
            username=username,
            tagline=tagline,
            font_name=font_name,
            background_info=background_info,
            color_schema=color_schema,
            image_provider=image_provider,
            image_model=image_model,
        )

        logger.info(
            f"Successfully generated infographic with reference image: {prompt}"
        )

        request_id = getattr(request.state, "request_id", None)
        return InfographicFromPromptResponse(
            success=True,
            status=result.get("status", "success"),
            prompt=result.get("prompt", prompt),
            type=result.get("type", "unknown"),
            title=result.get("title", ""),
            content=result.get("content", ""),
            image_path=result.get("image_path"),
            image_url=get_image_url(result.get("image_path"), request),
            image_filename=result.get("image_filename"),
            image_relative_path=result.get("image_relative_path"),
            request_id=request_id,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}",
        )
    except Exception as e:
        logger.error(
            f"Error generating infographic with reference image: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Infographic generation failed: {str(e)}",
        )


@app.get(
    "/",
    tags=["Root"],
    summary="API root endpoint",
)
async def root(request: Request):
    """
    Root endpoint with API information.

    Returns:
        API information and available endpoints
    """
    request_id = getattr(request.state, "request_id", None)
    return {
        "success": True,
        "name": "Social Media Content Generator API",
        "version": "1.0.0",
        "description": "Production-ready API for generating social media content (carousel posts and single informational images) from article URLs with AI-generated images",
        "endpoints": {
            "health": "/health",
            "generate": "/api/v1/generate",
            "generate_single_image": "/api/v1/generate-single-image",
            "generate_infographic_from_prompt": "/api/v1/generate-infographic-from-prompt",
            "generate_infographic_with_reference": "/api/v1/generate-infographic-with-reference",
            "serve_image": "/api/v1/images/{filename}",
            "docs": "/docs",
            "redoc": "/redoc",
        },
        "request_id": request_id,
    }


@app.get(
    "/api/v1/info",
    tags=["Info"],
    summary="API information endpoint",
)
async def api_info(request: Request):
    """
    Get detailed API information including capabilities and configuration.

    Returns:
        Detailed API information
    """
    request_id = getattr(request.state, "request_id", None)
    return {
        "success": True,
        "api": {
            "name": "Social Media Content Generator API",
            "version": "1.0.0",
            "description": "Production-ready API for generating social media content",
        },
        "capabilities": {
            "carousel_generation": True,
            "single_image_generation": True,
            "infographic_from_prompt": True,
            "infographic_with_reference": True,
            "image_serving": True,
        },
        "image_providers": ["openrouter", "openai"],
        "supported_models": {
            "openrouter": [
                "google/gemini-2.5-flash-image",
                "google/gemini-3-pro-image-preview",
                "x-ai/grok-4.1-fast",
            ],
            "openai": ["dall-e-3", "dall-e-2", "gpt-image-1", "gpt-image-1.5"],
        },
        "rate_limit": {
            "per_minute": RATE_LIMIT_PER_MINUTE,
        },
        "authentication": {
            "required": bool(API_KEY),
            "header": API_KEY_NAME if API_KEY else None,
        },
        "request_id": request_id,
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

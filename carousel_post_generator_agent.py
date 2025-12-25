"""
Standalone LangGraph Agent for Social Media Content Generation

This is a production-ready, standalone backend agent that uses LangGraph
to orchestrate social media content generation from article content.
Supports both carousel posts and single informational images.

INSTALLATION:
    pip install -r requirements.txt

REQUIRED ENVIRONMENT VARIABLES:
    - OPENAI_API_KEY: Your OpenAI API key (required for content generation)
    - OPENROUTER_API_KEY: Your OpenRouter API key (required for image generation)

    You can set them in:
    1. Environment variables: export VAR_NAME=your_value
    2. .env file in the same directory as this script

USAGE:
    python carousel_post_generator_agent.py

    Or as a module:
    python -m carousel_post_generator_agent

OUTPUT:
    Results are saved to the 'output' folder:
    - carousel_{sanitized_title}_{timestamp}.md: Carousel slides in markdown format
    - carousel_{sanitized_title}_{timestamp}.json: Carousel data in JSON format
"""

import logging
import json
import re
import time
import os
import base64
import traceback
from typing import TypedDict, Annotated, List, Optional, Dict, Any, Literal
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime
import asyncio

import aiohttp
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    try:
        script_dir = Path(__file__).parent.absolute()
        env_path = script_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
    except NameError:
        load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Image generation provider configuration
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "openrouter")  # "openrouter" or "openai"
IMAGE_MODEL = os.getenv(
    "IMAGE_MODEL", None
)  # Model name (e.g., "dall-e-3", "google/gemini-2.5-flash-image")


# ============================================================================
# State Definition
# ============================================================================


class SocialMediaContentState(TypedDict):
    """State for the social media content generator agent."""

    messages: Annotated[List, lambda x, y: x + y]
    url: str
    max_slides: int
    username: Optional[str]  # Social media username (e.g., "@robots")
    tagline: Optional[
        str
    ]  # Tagline/brand message (e.g., "daily programming tips & tricks")
    title: Optional[str]  # Custom title to override scraped article title
    extra_instructions: Optional[str]  # Additional instructions for the LLM
    font_name: Optional[str]  # Font name for slides (e.g., "Arial", "Roboto")
    background_info: Optional[str]  # Background description for slides
    color_schema: Optional[str]  # Color schema description for slides
    image_provider: Optional[
        str
    ]  # Image generation provider ("openrouter" or "openai")
    image_model: Optional[
        str
    ]  # Image generation model (e.g., "dall-e-3", "google/gemini-2.5-flash-image")
    output_folder: Optional[Path]  # Folder for saving images
    article_content: Optional[Dict[str, Any]]
    slides: Optional[List[Dict[str, Any]]]
    slides_with_images: Optional[List[Dict[str, Any]]]
    status: str
    error: Optional[str]


# ============================================================================
# Image Storage
# ============================================================================


def save_image_locally(
    image_data: bytes, output_folder: Path, slide_number: int, prompt: str
) -> Optional[Dict[str, str]]:
    """
    Save an image locally to the output folder.

    Args:
        image_data (bytes): The image data to save
        output_folder (Path): Folder to save the image in
        slide_number (int): Slide number for naming
        prompt (str): Prompt used for image generation (for filename)

    Returns:
        Optional[Dict[str, str]]: Dictionary with local file path, or None if save failed
    """
    try:
        # Create a safe filename from the prompt
        safe_prompt = re.sub(r"[^\w\s-]", "", prompt)[:30]
        safe_prompt = re.sub(r"[\s]+", "_", safe_prompt)

        # Create filename
        filename = f"slide_{slide_number}_{safe_prompt}.png"
        image_path = output_folder / filename

        # Save the image
        with open(image_path, "wb") as f:
            f.write(image_data)

        logger.info(f"Saved image to: {image_path}")
        return {
            "path": str(image_path),
            "filename": filename,
            "relative_path": f"./{filename}",
        }
    except Exception as e:
        logger.error(f"Error saving image locally: {e}")
        return None


# ============================================================================
# Web Scraping
# ============================================================================


async def scrape_article_content(url: str) -> Dict[str, Any]:
    """
    Scrape article content from a given URL using Playwright.

    Args:
        url: URL of the article to scrape

    Returns:
        Dictionary containing article title, content, and metadata
    """
    try:
        logger.info(f"Scraping article from: {url}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            try:
                page = await browser.new_page(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                )

                await page.goto(url, wait_until="networkidle", timeout=60000)
                await asyncio.sleep(2)

                # Extract article content
                content = await page.evaluate(
                    """
                    () => {
                        // Try to find the main article content
                        const article = document.querySelector('article') || 
                                      document.querySelector('main') ||
                                      document.querySelector('[role="main"]') ||
                                      document.body;
                        
                        // Get title
                        const title = document.querySelector('h1')?.textContent?.trim() || 
                                    document.title ||
                                    'Untitled Article';
                        
                        // Get meta description
                        const metaDesc = document.querySelector('meta[name="description"]')?.content || 
                                       document.querySelector('meta[property="og:description"]')?.content ||
                                       '';
                        
                        // Extract headings and paragraphs
                        const elements = article.querySelectorAll('h1, h2, h3, h4, h5, h6, p');
                        const content = Array.from(elements)
                            .map(el => ({
                                tag: el.tagName.toLowerCase(),
                                text: el.textContent?.trim()
                            }))
                            .filter(item => item.text && item.text.length > 0);
                        
                        // Get full text
                        const fullText = Array.from(article.querySelectorAll('p'))
                            .map(p => p.textContent?.trim())
                            .filter(text => text && text.length > 0)
                            .join('\\n\\n');
                        
                        return {
                            title,
                            metaDescription: metaDesc,
                            content: content,
                            fullText: fullText,
                            url: window.location.href
                        };
                    }
                    """
                )

                logger.info(
                    f"Successfully scraped article: {content.get('title', 'Unknown')}"
                )
                return content

            finally:
                await browser.close()

    except Exception as e:
        logger.error(f"Error scraping article: {str(e)}")
        raise


# ============================================================================
# Image Generation (OpenRouter Gemini & OpenAI DALL-E)
# ============================================================================


def generate_image_with_openai_dalle(
    prompt: str,
    slide_number: int,
    output_folder: Path,
    orientation: str = "square",
    model: str = "dall-e-3",
    size: str = "1024x1024",
    quality: str = "standard",
) -> Optional[Dict[str, str]]:
    """
    Generate an image using OpenAI image generation API (DALL-E or GPT-Image models).

    Args:
        prompt: Image generation prompt
        slide_number: Slide number for naming
        output_folder: Folder to save the image in
        orientation: Image orientation (square recommended for carousels)
        model: Model to use ("dall-e-3", "dall-e-2", "gpt-image-1.5", etc.)
        size: Image size ("1024x1024", "1792x1024", "1024x1792", "1536x1024", "1024x1536")
        quality: Image quality ("standard", "hd", "high", "medium", "low", or "auto")

    Returns:
        Dictionary with local image path and metadata, or None if failed
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        return None

    try:
        logger.info(f"Generating image with OpenAI {model} for slide {slide_number}")

        # OpenAI image generation API endpoint
        url = "https://api.openai.com/v1/images/generations"

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        # Determine size based on orientation and model
        if orientation == "square":
            image_size = "1024x1024"
        elif orientation == "vertical":
            # GPT-Image models support portrait sizes
            if model.startswith("gpt-image"):
                image_size = "1024x1536"
            elif model == "dall-e-3":
                image_size = "1024x1792"
            else:
                image_size = "1024x1024"
        else:
            image_size = size

        # Build payload
        payload = {
            "model": model,
            "prompt": prompt,
            "size": image_size,
            "n": 1,
        }

        # Add quality parameter based on model
        if model == "dall-e-3":
            payload["quality"] = quality
        elif model.startswith("gpt-image"):
            # GPT-Image models support: "high", "medium", "low", or "auto"
            if quality in ["high", "medium", "low", "auto"]:
                payload["quality"] = quality
            else:
                # Map DALL-E quality to GPT-Image quality
                quality_map = {"standard": "medium", "hd": "high"}
                payload["quality"] = quality_map.get(quality, "auto")

            # GPT-Image models always return base64, but we can request it explicitly
            # Note: response_format is only supported for dall-e-2, GPT-Image models always return base64
            # So we don't need to set response_format for GPT-Image models

        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code == 200:
            data = response.json()

            if "data" in data and len(data["data"]) > 0:
                image_data = data["data"][0]

                # Check for URL or base64 response
                if "url" in image_data:
                    # Download image from URL
                    image_url = image_data["url"]
                    img_response = requests.get(image_url, timeout=60)
                    if img_response.status_code == 200:
                        image_bytes = img_response.content
                    else:
                        logger.error(f"Failed to download image from URL: {image_url}")
                        return None
                elif "b64_json" in image_data:
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data["b64_json"])
                else:
                    logger.error("No image data found in OpenAI response")
                    return None

                # Save image locally
                save_result = save_image_locally(
                    image_bytes, output_folder, slide_number, prompt
                )

                if save_result:
                    logger.info(
                        f"Successfully generated and saved image with OpenAI {model} for slide {slide_number}"
                    )
                    return {
                        "path": save_result["path"],
                        "filename": save_result["filename"],
                        "relative_path": save_result["relative_path"],
                        "prompt": prompt,
                        "slide_number": slide_number,
                    }
                else:
                    logger.error("Failed to save image locally")
                    return None
            else:
                logger.error("No image data in OpenAI response")
                return None
        else:
            logger.error(
                f"OpenAI API returned status code {response.status_code}: {response.text}"
            )
            return None

    except Exception as e:
        logger.error(f"Error generating image with OpenAI {model}: {e}")
        logger.error(traceback.format_exc())
        return None


def generate_image_with_openai_edits(
    prompt: str,
    slide_number: int,
    output_folder: Path,
    reference_image_base64: str,
    orientation: str = "square",
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    quality: str = "auto",
) -> Optional[Dict[str, str]]:
    """
    Generate an image using OpenAI images/edits endpoint with reference image (GPT-Image models).

    Args:
        prompt: Image generation prompt
        slide_number: Slide number for naming
        output_folder: Folder to save the image in
        reference_image_base64: Base64-encoded reference image to use as design guide
        orientation: Image orientation (square recommended for carousels)
        model: GPT-Image model to use ("gpt-image-1", "gpt-image-1.5", etc.)
        size: Image size ("1024x1024", "1536x1024", "1024x1536", or "auto")
        quality: Image quality ("high", "medium", "low", or "auto")

    Returns:
        Dictionary with local image path and metadata, or None if failed
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        return None

    try:
        logger.info(
            f"Generating image with OpenAI {model} (edits endpoint) for slide {slide_number} using reference image"
        )

        # OpenAI images/edits API endpoint (supports reference images)
        url = "https://api.openai.com/v1/images/edits"

        # Determine size based on orientation
        if orientation == "square":
            image_size = "1024x1024"
        elif orientation == "vertical":
            image_size = "1024x1536"
        else:
            image_size = size

        # Decode base64 reference image to bytes
        try:
            reference_image_bytes = base64.b64decode(reference_image_base64)
        except Exception as e:
            logger.error(f"Failed to decode reference image base64: {e}")
            return None

        # Prepare multipart/form-data payload
        # The image parameter accepts a single file or array of files
        # For requests library with multipart/form-data, we use a tuple: (filename, file_bytes, content_type)
        files = {"image": ("reference.png", reference_image_bytes, "image/png")}

        data = {
            "model": model,
            "prompt": prompt,
            "size": image_size,
            "n": 1,
        }

        # Add quality parameter for GPT-Image models
        if quality in ["high", "medium", "low", "auto"]:
            data["quality"] = quality
        else:
            data["quality"] = "auto"

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }

        response = requests.post(
            url, headers=headers, files=files, data=data, timeout=120
        )

        if response.status_code == 200:
            result = response.json()

            if "data" in result and len(result["data"]) > 0:
                image_data = result["data"][0]

                # GPT-Image models always return base64
                if "b64_json" in image_data:
                    image_bytes = base64.b64decode(image_data["b64_json"])
                elif "url" in image_data:
                    # Fallback to URL if provided
                    image_url = image_data["url"]
                    img_response = requests.get(image_url, timeout=60)
                    if img_response.status_code == 200:
                        image_bytes = img_response.content
                    else:
                        logger.error(f"Failed to download image from URL: {image_url}")
                        return None
                else:
                    logger.error("No image data found in OpenAI response")
                    return None

                # Save image locally
                save_result = save_image_locally(
                    image_bytes, output_folder, slide_number, prompt
                )

                if save_result:
                    logger.info(
                        f"Successfully generated and saved image with OpenAI {model} (edits) for slide {slide_number}"
                    )
                    return {
                        "path": save_result["path"],
                        "filename": save_result["filename"],
                        "relative_path": save_result["relative_path"],
                        "prompt": prompt,
                        "slide_number": slide_number,
                    }
                else:
                    logger.error("Failed to save image locally")
                    return None
            else:
                logger.error("No image data in OpenAI response")
                return None
        else:
            logger.error(
                f"OpenAI API returned status code {response.status_code}: {response.text}"
            )
            return None

    except Exception as e:
        logger.error(f"Error generating image with OpenAI {model} (edits): {e}")
        logger.error(traceback.format_exc())
        return None


def generate_carousel_image(
    prompt: str,
    slide_number: int,
    output_folder: Path,
    orientation: str = "square",
    reference_image_base64: Optional[str] = None,
    provider: str = "openrouter",
    model: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """
    Generate an image for a carousel slide using OpenRouter or OpenAI API and save locally.

    Args:
        prompt: Image generation prompt
        slide_number: Slide number for naming
        output_folder: Folder to save the image in
        orientation: Image orientation (square recommended for carousels)
        reference_image_base64: Optional base64-encoded reference image to use as design guide
        provider: Image generation provider ("openrouter" or "openai")
        model: Model name (e.g., "dall-e-3", "dall-e-2", "gpt-image-1", "gpt-image-1.5", "google/gemini-2.5-flash-image")
               If None, uses default based on provider

    Returns:
        Dictionary with local image path and metadata, or None if failed
    """
    # Route to OpenAI if provider is "openai"
    if provider.lower() == "openai":
        openai_model = model or "dall-e-3"

        # GPT-Image models support reference images via the edits endpoint
        if reference_image_base64 and openai_model.startswith("gpt-image"):
            logger.info(
                f"Using OpenAI {openai_model} edits endpoint with reference image"
            )
            return generate_image_with_openai_edits(
                prompt=prompt,
                slide_number=slide_number,
                output_folder=output_folder,
                reference_image_base64=reference_image_base64,
                orientation=orientation,
                model=openai_model,
            )
        # DALL-E models don't support reference images
        elif reference_image_base64:
            logger.warning(
                f"OpenAI {openai_model} doesn't support reference images. "
                "Reference image will be ignored. "
                "Consider using 'gpt-image-1' or 'gpt-image-1.5' for reference image support, "
                "or use 'openrouter' provider with a vision model."
            )

        # Use generations endpoint for DALL-E or GPT-Image without reference
        return generate_image_with_openai_dalle(
            prompt=prompt,
            slide_number=slide_number,
            output_folder=output_folder,
            orientation=orientation,
            model=openai_model,
        )

    # Default to OpenRouter (existing logic)
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not found in environment variables")
        return None

    try:
        # Create carousel-optimized prompt
        if reference_image_base64:
            design_instruction = f"""CRITICAL INSTRUCTION: You MUST create a new image that is an EXACT REPLICA of the attached reference image's design, but with NEW CONTENT.

REPLICATE EXACTLY FROM THE REFERENCE IMAGE:
- SAME background color/gradient (copy the exact colors)
- SAME card/box layout and grid structure (copy the exact layout)
- SAME header bar style, shape, and colors
- SAME card shapes, corner radius, and shadows
- SAME typography style, font weights, and text colors
- SAME icon placement and style
- SAME spacing, margins, and padding between elements
- SAME overall visual hierarchy and structure

ONLY CHANGE THE TEXT CONTENT TO:
{prompt}

The final image should look like it was made from the same template as the reference image - a viewer should not be able to tell the difference in design style. Only the text content should be different.

Generate a {orientation} infographic that is visually IDENTICAL to the reference image in terms of design, layout, and style."""
        else:
            design_instruction = f"Create a {orientation} carousel slide image: {prompt}. Make it clean, modern, and visually engaging with clear focal point. Perfect for social media carousel."

        logger.info(f"Generating image for slide {slide_number}: {prompt}")

        # OpenRouter API endpoint
        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://contentrob.com",
            "X-Title": "ContentRob",
        }

        # Build message content
        if reference_image_base64:
            # Include reference image in the message
            message_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{reference_image_base64}"
                    },
                },
                {"type": "text", "text": design_instruction},
            ]
        else:
            message_content = design_instruction

        # Use provided model or default based on provider
        openrouter_model = model or "google/gemini-2.5-flash-image"

        payload = {
            "model": openrouter_model,
            "messages": [{"role": "user", "content": message_content}],
            "modalities": ["image", "text"],
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code == 200:
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]

                # Check for errors in the response (some APIs return 200 with error in body)
                if "error" in choice:
                    error_info = choice["error"]
                    error_message = error_info.get("message", "Unknown error")
                    error_code = error_info.get("code", "unknown")
                    logger.error(
                        f"API returned error in response: {error_message} (code: {error_code})"
                    )
                    return None

                message = choice.get("message", {})

                # Check if message content is empty (indicates an error or incomplete response)
                if not message.get("content") and not message.get("images"):
                    logger.error(
                        f"Empty response from API. Finish reason: {choice.get('finish_reason', 'unknown')}"
                    )
                    return None

                # Check for image response (Gemini format)
                if "images" in message and len(message["images"]) > 0:
                    image_data = message["images"][0]

                    if "image_url" in image_data and "url" in image_data["image_url"]:
                        image_url = image_data["image_url"]["url"]

                        if image_url.startswith("data:image/"):
                            # Extract base64 data
                            header, base64_data = image_url.split(",", 1)
                            image_bytes = base64.b64decode(base64_data)

                            # Save image locally
                            save_result = save_image_locally(
                                image_bytes, output_folder, slide_number, prompt
                            )

                            if save_result:
                                logger.info(
                                    f"Successfully generated and saved image for slide {slide_number}"
                                )
                                return {
                                    "path": save_result["path"],
                                    "filename": save_result["filename"],
                                    "relative_path": save_result["relative_path"],
                                    "prompt": prompt,
                                    "slide_number": slide_number,
                                }
                            else:
                                logger.error("Failed to save image locally")
                                return None

                # Check for SVG/text response (Grok and other text-based models)
                content = message.get("content", "")

                # Skip if content is empty (error case already handled above)
                if not content or not content.strip():
                    logger.error(
                        f"No content in response. Finish reason: {choice.get('finish_reason', 'unknown')}"
                    )
                    return None

                if content and (
                    "<svg" in content.lower()
                    or "```svg" in content.lower()
                    or "svg" in content.lower()
                ):
                    logger.info(
                        f"Detected SVG response from model, converting to PNG..."
                    )

                    # Extract SVG code from markdown code blocks if present
                    svg_content = content

                    # First, try to extract from ```svg code block
                    if "```svg" in content.lower():
                        svg_match = re.search(
                            r"```svg\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE
                        )
                        if svg_match:
                            svg_content = svg_match.group(1).strip()
                    # Then try generic code block that might contain SVG
                    elif "```" in content and "<svg" in content.lower():
                        svg_match = re.search(
                            r"```[^\n]*\s*(.*?)\s*```", content, re.DOTALL
                        )
                        if svg_match:
                            potential_svg = svg_match.group(1).strip()
                            if "<svg" in potential_svg.lower():
                                svg_content = potential_svg

                    # Extract SVG tag if it's embedded in text (handles cases where SVG is in text without code blocks)
                    if "<svg" in svg_content.lower():
                        svg_match = re.search(
                            r"<svg.*?</svg>", svg_content, re.DOTALL | re.IGNORECASE
                        )
                        if svg_match:
                            svg_content = svg_match.group(0)

                    # Convert SVG to PNG using cairosvg or fallback method
                    try:
                        import cairosvg

                        image_bytes = cairosvg.svg2png(
                            bytestring=svg_content.encode("utf-8")
                        )

                        # Save image locally
                        save_result = save_image_locally(
                            image_bytes, output_folder, slide_number, prompt
                        )

                        if save_result:
                            logger.info(
                                f"Successfully converted SVG and saved image for slide {slide_number}"
                            )
                            return {
                                "path": save_result["path"],
                                "filename": save_result["filename"],
                                "relative_path": save_result["relative_path"],
                                "prompt": prompt,
                                "slide_number": slide_number,
                            }
                        else:
                            logger.error("Failed to save converted SVG image locally")
                            return None
                    except ImportError:
                        logger.error(
                            "cairosvg not installed. Install it with: pip install cairosvg"
                        )
                        logger.error(
                            "Grok and other text-based models return SVG code. "
                            "Please install cairosvg to convert SVG to images, "
                            "or use a model that generates images directly (e.g., google/gemini-2.5-flash-image)"
                        )
                        return None
                    except Exception as e:
                        logger.error(f"Error converting SVG to PNG: {e}")
                        logger.error(traceback.format_exc())
                        return None
                else:
                    # Content exists but doesn't contain SVG - log for debugging
                    logger.warning(
                        f"Response contains content but no SVG detected. Content preview: {content[:200]}..."
                    )
                    logger.error(
                        f"Unable to process response. Expected image or SVG format. "
                        f"Finish reason: {choice.get('finish_reason', 'unknown')}"
                    )
                    return None
            else:
                logger.error("No choices in API response")
                return None
        else:
            logger.error(
                f"OpenRouter API returned status code {response.status_code}: {response.text}"
            )
            return None

    except Exception as e:
        logger.error(f"Error generating image: {e}")
        logger.error(traceback.format_exc())
        return None


# ============================================================================
# LLM Service
# ============================================================================


class LLMService:
    """Service for generating carousel slide content using LLM."""

    def __init__(self, model_name: str = "gpt-5.2-2025-12-11", temperature: float = 1):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
        )

    def generate_carousel_slides(
        self,
        article_content: Dict[str, Any],
        max_slides: int = 10,
        username: Optional[str] = None,
        tagline: Optional[str] = None,
        title: Optional[str] = None,
        extra_instructions: Optional[str] = None,
        font_name: Optional[str] = None,
        background_info: Optional[str] = None,
        color_schema: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate carousel slide content from article.

        Args:
            article_content: Scraped article content
            max_slides: Maximum number of slides to generate
            username: Social media username (e.g., "@robots")
            tagline: Tagline/brand message (e.g., "daily programming tips & tricks")
            title: Custom title to override scraped article title
            extra_instructions: Additional instructions for the LLM

        Returns:
            List of slide dictionaries with title, content, and image prompt
        """
        try:
            # Use custom title if provided, otherwise use scraped title
            title = title or article_content.get("title", "Unknown")
            full_text = article_content.get("fullText", "")
            meta_desc = article_content.get("metaDescription", "")

            #             prompt = f"""
            # You are an expert social media content creator specializing in carousel posts.

            # Given the following article, create a engaging carousel post with {max_slides} slides.

            # Article Title: {title}
            # Meta Description: {meta_desc}
            # Article Content:
            # {full_text[:5000]}

            # Create {max_slides} carousel slides. Each slide should:
            # 1. Have a clear, attention-grabbing title (max 60 characters)
            # 2. Have concise, valuable content (2-3 short sentences or bullet points, max 150 characters)
            # 3. Include a detailed image generation prompt describing what visual would best illustrate the slide

            # Guidelines:
            # - Slide 1 should be a hook/title slide
            # - Middle slides should cover key points from the article
            # - Last slide should have a call-to-action or conclusion
            # - Keep text minimal and impactful
            # - Make each slide self-contained but part of a cohesive story

            # Return ONLY a JSON array with this exact structure:
            # [
            #   {{
            #     "slide_number": 1,
            #     "title": "Catchy Title Here",
            #     "content": "Brief, impactful content here.",
            #     "image_prompt": "Detailed description for image generation"
            #   }}
            # ]

            # Respond ONLY with the JSON array, no other text.
            # """
            # Set defaults for username and tagline
            display_username = username or "@coding180.com"
            display_tagline = tagline or "Daily AI Tools & Agents"

            # Set defaults for design parameters
            display_font_name = font_name or "modern sans-serif font"
            display_background = (
                background_info
                or "Clean gradient or subtle tech/coding theme (dark navy/blue/purple or modern light mode)"
            )
            display_color_schema = (
                color_schema
                or "Consistent across all slides (e.g., navy background, white + cyan accent text)"
            )

            # Build extra instructions section if provided
            extra_instructions_section = ""
            if extra_instructions:
                extra_instructions_section = (
                    f"\n=== ADDITIONAL INSTRUCTIONS ===\n{extra_instructions}\n"
                )

            prompt = f"""
You are an expert LinkedIn/Instagram carousel designer who creates HIGHLY ENGAGING informational carousels that get thousands of saves and shares.

Your job: Turn the article below into exactly {max_slides} beautiful, text-on-image carousel slides.
Every slide image must contain the text (title + content) directly on the image — no separate caption text, no illustrative-only images.

=== STRICT SLIDE STRUCTURE ===
- Slide 1: Hook / Introduction slide
  - Big catchy title (the article title or a punchy version)
  - 1–2 sentence teaser
  - End with "Swipe →" or "Keep reading →"
- Slides 2 to {max_slides-1}: Content slides
  - Summarize and group the article's main points logically
  - If the article has 10 tips and we have only 5 slides total → group 2–3 tips per slide
  - Use short, scannable bullet points
  - Never cut important information — condense wisely
- Slide {max_slides}: Final CTA slide
  - Short recap or strongest takeaway
  - Big call-to-action: "Save this carousel for later!", "Which tip will you try first?", "Tag a friend who needs this!"
  - Prominent text: "Follow {display_username} for {display_tagline} →"
  - Optional: your logo or handle in the corner

=== DESIGN RULES FOR EVERY IMAGE_PROMPT (CRITICAL) ===
All slides must look like professional Canva-style carousel slides:
- Format: Square or vertical orientation (1080x1080 or 1080x1350 aspect ratio)
- Background: {display_background}
- Color scheme: {display_color_schema}
- Title: Extra large bold {display_font_name}, top portion of slide
- Body text: Clean bullet points, highly readable, max 7 lines
- Add "Slide X of {max_slides}" in small text at top-right or bottom-right
- Add subtle relevant icons (code symbols, laptop, lightbulb, rocket, etc.)
- Add small "{display_username}" handle in bottom-left or bottom-right corner on every slide
- High contrast, modern, premium feel — looks expensive
- DO NOT include technical specifications like pixel sizes, font names, font sizes, or hex color codes in the image prompts

=== OUTPUT FORMAT ===
Return ONLY a valid JSON array (no markdown, no explanation). Each object must have exactly these keys:

[
  {{
    "slide_number": 1,
    "title": "Exact title text that will appear on the image",
    "content": "Exact body text that will appear on the image (use \\n for line breaks in bullets)",
    "image_prompt": "Extremely detailed prompt that forces Gemini to render the exact title and content as text on the image. Include layout, colors, fonts, and all text verbatim. Do NOT include pixel sizes, font names, font sizes, or hex color codes."
  }}
]
{extra_instructions_section}
Article Title: {title}
Meta Description: {meta_desc}
Full Article Text:
{full_text}

Now generate exactly {max_slides} slides following all rules above.
"""

            logger.info("Generating carousel slides with LLM...")
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content.strip()

            # Extract JSON from response
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = re.sub(r"```json\s*|\s*```", "", response_text).strip()

            slides = json.loads(response_text)

            logger.info(f"Successfully generated {len(slides)} carousel slides")
            return slides

        except Exception as e:
            logger.error(f"Error generating carousel slides: {str(e)}")
            raise

    def generate_carousel_from_prompt(
        self,
        user_prompt: str,
        max_slides: int = 10,
        username: Optional[str] = None,
        tagline: Optional[str] = None,
        font_name: Optional[str] = None,
        background_info: Optional[str] = None,
        color_schema: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate carousel slide content from a user text prompt.

        Args:
            user_prompt: User's text prompt (e.g., "top 10 free AI tools", "how to learn React")
            max_slides: Maximum number of slides to generate
            username: Social media username (e.g., "@robots")
            tagline: Tagline/brand message (e.g., "daily programming tips & tricks")
            font_name: Font name for the carousel slides
            background_info: Background description
            color_schema: Color schema description
            extra_instructions: Additional instructions for the LLM

        Returns:
            List of slide dictionaries with title, content, and image prompt
        """
        try:
            # Set defaults for username and tagline
            display_username = username or "@coding180.com"
            display_tagline = tagline or "Daily AI Tools & Agents"

            # Set defaults for design parameters
            display_font_name = font_name or "modern sans-serif font"
            display_background = (
                background_info
                or "Clean gradient or subtle tech/coding theme (dark navy/blue/purple or modern light mode)"
            )
            display_color_schema = (
                color_schema
                or "Consistent across all slides (e.g., navy background, white + cyan accent text)"
            )

            # Build extra instructions section if provided
            extra_instructions_section = ""
            if extra_instructions:
                extra_instructions_section = (
                    f"\n=== ADDITIONAL INSTRUCTIONS ===\n{extra_instructions}\n"
                )

            prompt = f"""
You are an expert LinkedIn/Instagram carousel designer who creates HIGHLY ENGAGING informational carousels that get thousands of saves and shares.

Your job: Create exactly {max_slides} beautiful, text-on-image carousel slides based on the user's prompt below.
Every slide image must contain the text (title + content) directly on the image — no separate caption text, no illustrative-only images.

=== PROMPT ANALYSIS ===
Analyze the user's prompt to understand what content to create:
- LIST TYPE: Prompts asking for lists, rankings, collections (e.g., "top 10 AI tools", "5 best practices", "7 ways to...")
  - Generate a comprehensive, accurate list based on current knowledge
  - Extract or create the list items (numbered or bulleted)
  - Ensure items are relevant, accurate, and valuable
  - If the prompt mentions a specific number (e.g., "top 10"), generate exactly that many items
  - Distribute items across slides logically (e.g., 2-3 items per slide for a 10-item list)

- GUIDE/TUTORIAL TYPE: Prompts asking for explanations, concepts, guides, tips (e.g., "how to learn React", "explain machine learning", "benefits of TypeScript")
  - Create a comprehensive guide with key points
  - Extract main concepts, benefits, or key takeaways
  - Organize into logical sections across slides
  - Each slide should cover a distinct concept or step

=== STRICT SLIDE STRUCTURE ===
- Slide 1: Hook / Introduction slide
  - Big catchy title (derived from the user prompt)
  - 1–2 sentence teaser explaining what the carousel covers
  - End with "Swipe →" or "Keep reading →"
- Slides 2 to {max_slides-1}: Content slides
  - Distribute the main content logically across these slides
  - Use short, scannable bullet points
  - Each slide should be self-contained but part of a cohesive story
  - Group related points together
- Slide {max_slides}: Final CTA slide
  - Short recap or strongest takeaway
  - Big call-to-action: "Save this carousel for later!", "Which tip will you try first?", "Tag a friend who needs this!"
  - Prominent text: "Follow {display_username} for {display_tagline} →"
  - Optional: your logo or handle in the corner

=== CONTENT GENERATION RULES ===
- Be accurate, helpful, and current
- Use clear, concise language
- Prioritize the most valuable information
- For lists: Ensure all items are relevant to the prompt
- For guides: Focus on the most important concepts and steps
- Make content actionable and valuable

=== DESIGN RULES FOR EVERY IMAGE_PROMPT (CRITICAL) ===
All slides must look like professional Canva-style carousel slides:
- Format: Square or vertical orientation (1080x1080 or 1080x1350 aspect ratio)
- Background: {display_background}
- Color scheme: {display_color_schema}
- Title: Extra large bold {display_font_name}, top portion of slide
- Body text: Clean bullet points, highly readable, max 7 lines
- Add "Slide X of {max_slides}" in small text at top-right or bottom-right
- Add subtle relevant icons (code symbols, laptop, lightbulb, rocket, etc.)
- Add small "{display_username}" handle in bottom-left or bottom-right corner on every slide
- High contrast, modern, premium feel — looks expensive
- DO NOT include technical specifications like pixel sizes, font names, font sizes, or hex color codes in the image prompts

=== OUTPUT FORMAT ===
Return ONLY a valid JSON array (no markdown, no explanation). Each object must have exactly these keys:

[
  {{
    "slide_number": 1,
    "title": "Exact title text that will appear on the image",
    "content": "Exact body text that will appear on the image (use \\n for line breaks in bullets)",
    "image_prompt": "Extremely detailed prompt that forces Gemini to render the exact title and content as text on the image. Include layout, colors, fonts, and all text verbatim. Do NOT include pixel sizes, font names, font sizes, or hex color codes."
  }}
]
{extra_instructions_section}
User Prompt: {user_prompt}

Now generate exactly {max_slides} slides following all rules above.
"""

            logger.info(f"Generating carousel slides from prompt: {user_prompt}")
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content.strip()

            # Extract JSON from response
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = re.sub(r"```json\s*|\s*```", "", response_text).strip()

            slides = json.loads(response_text)

            logger.info(
                f"Successfully generated {len(slides)} carousel slides from prompt"
            )
            return slides

        except Exception as e:
            logger.error(f"Error generating carousel slides from prompt: {str(e)}")
            raise

    def generate_single_informational_image(
        self,
        article_content: Dict[str, Any],
        username: Optional[str] = None,
        tagline: Optional[str] = None,
        title: Optional[str] = None,
        extra_instructions: Optional[str] = None,
        font_name: Optional[str] = None,
        background_info: Optional[str] = None,
        color_schema: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a single informational image from article.
        - If list post: extracts list items and creates one image with the list
        - If general article: summarizes and creates one image

        Args:
            article_content: Scraped article content
            username: Social media username (e.g., "@robots")
            tagline: Tagline/brand message (e.g., "daily programming tips & tricks")
            title: Custom title to override scraped article title
            extra_instructions: Additional instructions for the LLM

        Returns:
            Dictionary with title, content, and image prompt
        """
        try:
            # Use custom title if provided, otherwise use scraped title
            title = title or article_content.get("title", "Unknown")
            full_text = article_content.get("fullText", "")
            meta_desc = article_content.get("metaDescription", "")

            # Set defaults for username and tagline
            display_username = username or "@coding_robort"
            display_tagline = tagline or "daily programming tips & tricks"

            # Set defaults for design parameters
            display_font_name = font_name or "modern sans-serif font"
            display_background = (
                background_info
                or "Clean gradient or subtle tech/coding theme (dark navy/blue/purple or modern light mode)"
            )
            display_color_schema = (
                color_schema
                or "Modern and visually appealing (e.g., navy background, white + cyan accent text)"
            )

            # Build extra instructions section if provided
            extra_instructions_section = ""
            if extra_instructions:
                extra_instructions_section = (
                    f"\n=== ADDITIONAL INSTRUCTIONS ===\n{extra_instructions}\n"
                )

            prompt = f"""
You are an expert social media content designer who creates HIGHLY ENGAGING informational images.

Your job: Analyze the article below and create a SINGLE informational image that summarizes the key content.

=== ANALYSIS INSTRUCTIONS ===
1. First, determine if this is a LIST POST or GENERAL ARTICLE:
   - LIST POST: Contains numbered/bulleted lists, "Top X", "Best X", "X Ways to", "X Tips", etc.
   - GENERAL ARTICLE: Regular article with paragraphs, explanations, no clear list structure

2. If LIST POST:
   - Extract ALL list items (numbered or bulleted)
   - Create a single image with:
     * Main title (the article title or a punchy version)
     * All list items clearly displayed (numbered or bulleted)
     * Clean, organized layout that fits all items
   - Keep each list item concise (1-2 lines max)
   - If there are too many items (more than 10), select the most important ones

3. If GENERAL ARTICLE:
   - Create a comprehensive summary image with:
     * Main title (the article title or a punchy version)
     * Key takeaways (3-5 main points as bullet points)
     * Brief summary paragraph (2-3 sentences)
   - Focus on the most valuable information

=== DESIGN RULES FOR IMAGE_PROMPT (CRITICAL) ===
The image must look like a professional informational graphic:
- Format: Square or vertical orientation (1080x1080 or 1080x1350 aspect ratio)
- Background: {display_background}
- Color scheme: {display_color_schema}
- Title: Extra large bold {display_font_name}, top portion of image
- Content: Clean, organized layout with clear hierarchy
- Add small "{display_username}" handle in bottom-left or bottom-right corner
- High contrast, modern, premium feel — looks expensive
- DO NOT include technical specifications like pixel sizes, font names, font sizes, or hex color codes in the image prompts

=== OUTPUT FORMAT ===
Return ONLY a valid JSON object (no markdown, no explanation) with exactly these keys:

{{
  "is_list_post": true/false,
  "title": "Exact title text that will appear on the image",
  "content": "Exact content that will appear on the image (use \\n for line breaks, numbered or bulleted format)",
  "image_prompt": "Extremely detailed prompt that forces Gemini to render the exact title and content as text on the image. Include layout, colors, fonts, and all text verbatim. Do NOT include pixel sizes, font names, font sizes, or hex color codes."
}}
{extra_instructions_section}
Article Title: {title}
Meta Description: {meta_desc}
Full Article Text:
{full_text}

Now analyze the article and generate the single informational image following all rules above.
"""

            logger.info("Generating single informational image with LLM...")
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content.strip()

            # Extract JSON from response
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = re.sub(r"```json\s*|\s*```", "", response_text).strip()

            result = json.loads(response_text)

            logger.info(
                f"Successfully generated single informational image (is_list_post: {result.get('is_list_post', False)})"
            )
            return result

        except Exception as e:
            logger.error(f"Error generating single informational image: {str(e)}")
            raise

    def generate_infographic_from_prompt(
        self,
        user_prompt: str,
        username: Optional[str] = None,
        tagline: Optional[str] = None,
        font_name: Optional[str] = None,
        background_info: Optional[str] = None,
        color_schema: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an infographic from a user text prompt.
        - Analyzes the prompt to determine if it's a list type (e.g., "top 10 tools") or summary type
        - If list: Generates the list items based on the prompt
        - If summary: Creates a summary infographic
        - Then generates the infographic image

        Args:
            user_prompt: User's text prompt (e.g., "top 10 free AI tools")
            username: Social media username (e.g., "@robots")
            tagline: Tagline/brand message (e.g., "daily programming tips & tricks")
            font_name: Font name for the infographic
            background_info: Background description
            color_schema: Color schema description
            extra_instructions: Additional instructions for the LLM

        Returns:
            Dictionary with title, content, image prompt, and type (list/summary)
        """
        try:
            # Set defaults for username and tagline
            display_username = username or "@coding_robort"
            display_tagline = tagline or "daily programming tips & tricks"

            # Set defaults for design parameters
            display_font_name = font_name or "modern sans-serif font"
            display_background = (
                background_info
                or "Clean gradient or subtle tech/coding theme (dark navy/blue/purple or modern light mode)"
            )
            display_color_schema = (
                color_schema
                or "Modern and visually appealing (e.g., navy background, white + cyan accent text)"
            )

            # Build extra instructions section if provided
            extra_instructions_section = ""
            if extra_instructions:
                extra_instructions_section = (
                    f"\n=== ADDITIONAL INSTRUCTIONS ===\n{extra_instructions}\n"
                )

            prompt = f"""
You are an expert social media content designer who creates HIGHLY ENGAGING informational infographics from user prompts.

Your job: Analyze the user's prompt below and create a SINGLE informational infographic.

=== PROMPT ANALYSIS ===
First, analyze the user's prompt to determine the type:
1. LIST TYPE: Prompts asking for lists, rankings, collections (e.g., "top 10 AI tools", "5 best practices", "7 ways to...", "best free tools")
   - For LIST TYPE: Generate a comprehensive, accurate list based on current knowledge
   - Extract or create the list items (numbered or bulleted)
   - Ensure items are relevant, accurate, and valuable
   - If the prompt mentions a specific number (e.g., "top 10"), generate exactly that many items
   - Each item should be concise but informative (1-2 lines max)

2. SUMMARY TYPE: Prompts asking for explanations, concepts, guides, tips (e.g., "how to use React hooks", "explain machine learning", "benefits of TypeScript")
   - For SUMMARY TYPE: Create a comprehensive summary with key points
   - Extract main concepts, benefits, or key takeaways
   - Organize into 3-7 main points as bullet points
   - Add a brief summary paragraph (2-3 sentences)

=== CONTENT GENERATION RULES ===
- Be accurate, helpful, and current
- Use clear, concise language
- Prioritize the most valuable information
- For lists: Ensure all items are relevant to the prompt
- For summaries: Focus on the most important concepts

=== DESIGN RULES FOR IMAGE_PROMPT (CRITICAL) ===
The image must look like a professional informational graphic:
- Format: Square or vertical orientation (1080x1080 or 1080x1350 aspect ratio)
- Background: {display_background}
- Color scheme: {display_color_schema}
- Title: Extra large bold {display_font_name}, top portion of image
- Content: Clean, organized layout with clear hierarchy
- For lists: Use numbered or bulleted format, clearly organized
- For summaries: Use bullet points for key takeaways, organized sections
- Add small "{display_username}" handle in bottom-left or bottom-right corner
- High contrast, modern, premium feel — looks expensive
- DO NOT include technical specifications like pixel sizes, font names, font sizes, or hex color codes in the image prompts

=== OUTPUT FORMAT ===
Return ONLY a valid JSON object (no markdown, no explanation) with exactly these keys:

{{
  "type": "list" or "summary",
  "title": "Exact title text that will appear on the image (derived from or matching the user prompt)",
  "content": "Exact content that will appear on the image (use \\n for line breaks, numbered or bulleted format)",
  "image_prompt": "Extremely detailed prompt that forces Gemini to render the exact title and content as text on the image. Include layout, colors, fonts, and all text verbatim. Do NOT include pixel sizes, font names, font sizes, or hex color codes."
}}
{extra_instructions_section}
User Prompt: {user_prompt}

Now analyze the prompt, generate the appropriate content (list or summary), and create the infographic following all rules above.
"""

            logger.info(f"Generating infographic from prompt: {user_prompt}")
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content.strip()

            # Extract JSON from response
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = re.sub(r"```json\s*|\s*```", "", response_text).strip()

            result = json.loads(response_text)

            logger.info(
                f"Successfully generated infographic from prompt (type: {result.get('type', 'unknown')})"
            )
            return result

        except Exception as e:
            logger.error(f"Error generating infographic from prompt: {str(e)}")
            raise


# ============================================================================
# Graph Nodes
# ============================================================================


async def scrape_article_node(
    state: SocialMediaContentState,
) -> SocialMediaContentState:
    """Scrape article content from URL."""
    try:
        print("\n" + "=" * 80)
        print("🔍 NODE: scrape_article_node - Scraping article")
        print("=" * 80)
        logger.info(f"Scraping article from URL: {state['url']}")

        article_content = await scrape_article_content(state["url"])

        if not article_content or not article_content.get("fullText"):
            print("❌ No content found")
            return {
                **state,
                "status": "error",
                "error": "No article content found",
            }

        # Create output folder based on article title
        article_title = article_content.get("title", "carousel_post")
        sanitized_title = sanitize_filename(article_title)

        try:
            script_dir = Path(__file__).parent.absolute()
            base_output_dir = script_dir / "output"
        except NameError:
            base_output_dir = Path.cwd() / "output"

        # Create a folder for this social media content
        content_folder = base_output_dir / sanitized_title
        content_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output folder: {content_folder}")

        print(f"✅ Successfully scraped article: {article_content.get('title')}")
        return {
            **state,
            "article_content": article_content,
            "output_folder": content_folder,
            "status": "scraped",
        }
    except Exception as e:
        print(f"❌ Error in scrape_article_node: {str(e)}")
        logger.error(f"Error in scrape_article_node: {str(e)}")
        return {**state, "status": "error", "error": str(e)}


async def generate_slides_node(
    state: SocialMediaContentState,
) -> SocialMediaContentState:
    """Generate carousel slide content using LLM."""
    try:
        print("\n" + "=" * 80)
        print("📝 NODE: generate_slides_node - Generating carousel slides")
        print("=" * 80)
        logger.info("Generating carousel slides")

        llm_service = LLMService()
        slides = llm_service.generate_carousel_slides(
            state["article_content"],
            state.get("max_slides", 10),
            username=state.get("username"),
            tagline=state.get("tagline"),
            title=state.get("title"),
            extra_instructions=state.get("extra_instructions"),
            font_name=state.get("font_name"),
            background_info=state.get("background_info"),
            color_schema=state.get("color_schema"),
        )

        if not slides:
            print("❌ No slides generated")
            return {**state, "status": "error", "error": "Failed to generate slides"}

        print(f"✅ Generated {len(slides)} carousel slides")
        return {
            **state,
            "slides": slides,
            "status": "slides_generated",
        }
    except Exception as e:
        print(f"❌ Error in generate_slides_node: {str(e)}")
        logger.error(f"Error in generate_slides_node: {str(e)}")
        return {**state, "status": "error", "error": str(e)}


async def generate_images_node(
    state: SocialMediaContentState,
) -> SocialMediaContentState:
    """Generate images for each carousel slide."""
    try:
        print("\n" + "=" * 80)
        print("🎨 NODE: generate_images_node - Generating carousel images")
        print("=" * 80)
        logger.info("Generating images for carousel slides")

        slides = state.get("slides", [])
        output_folder = state.get("output_folder")
        slides_with_images = []

        if not output_folder:
            logger.error("Output folder not set")
            return {**state, "status": "error", "error": "Output folder not set"}

        for slide in slides:
            slide_number = slide.get("slide_number", 0)
            image_prompt = slide.get("image_prompt", "")

            print(f"Generating image for slide {slide_number}...")

            # Generate image
            image_provider = state.get("image_provider") or IMAGE_PROVIDER
            image_model = state.get("image_model") or IMAGE_MODEL
            image_info = generate_carousel_image(
                image_prompt,
                slide_number,
                output_folder,
                provider=image_provider,
                model=image_model,
            )

            # Add image info to slide
            slide_with_image = {**slide}
            if image_info:
                slide_with_image["image_path"] = image_info["path"]
                slide_with_image["image_filename"] = image_info["filename"]
                slide_with_image["image_relative_path"] = image_info["relative_path"]
                print(f"✅ Image generated and saved for slide {slide_number}")
            else:
                slide_with_image["image_path"] = None
                slide_with_image["image_filename"] = None
                slide_with_image["image_relative_path"] = None
                print(f"⚠️  Failed to generate image for slide {slide_number}")

            slides_with_images.append(slide_with_image)

            # Small delay between image generations
            await asyncio.sleep(2)

        print(f"✅ Completed image generation for {len(slides_with_images)} slides")
        return {
            **state,
            "slides_with_images": slides_with_images,
            "status": "completed",
        }
    except Exception as e:
        print(f"❌ Error in generate_images_node: {str(e)}")
        logger.error(f"Error in generate_images_node: {str(e)}")
        return {**state, "status": "error", "error": str(e)}


async def agent_node(state: SocialMediaContentState) -> SocialMediaContentState:
    """Main agent node that orchestrates the workflow."""
    try:
        current_status = state.get("status", "initialized")

        if current_status == "initialized":
            return await scrape_article_node(state)

        elif current_status == "scraped":
            return await generate_slides_node(state)

        elif current_status == "slides_generated":
            return await generate_images_node(state)

        elif current_status == "completed":
            return state

        return state

    except Exception as e:
        logger.error(f"Error in agent_node: {str(e)}")
        return {**state, "status": "error", "error": str(e)}


def should_continue(state: SocialMediaContentState) -> Literal["continue", "end"]:
    """Determine if the workflow should continue or end."""
    status = state.get("status", "")

    if status == "completed":
        return "end"
    elif status == "error":
        return "end"
    else:
        return "continue"


# ============================================================================
# Graph Construction
# ============================================================================


def create_social_media_content_generator_agent():
    """Create and compile the social media content generator LangGraph agent."""

    workflow = StateGraph(SocialMediaContentState)

    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "agent",
            "end": END,
        },
    )

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph


# ============================================================================
# Agent Interface
# ============================================================================


class SocialMediaContentGeneratorAgent:
    """Standalone LangGraph agent for social media content generation."""

    def __init__(self):
        self.graph = create_social_media_content_generator_agent()
        logger.info("Social Media Content Generator Agent initialized")

    async def process(
        self,
        url: str,
        max_slides: int = 10,
        username: Optional[str] = None,
        tagline: Optional[str] = None,
        title: Optional[str] = None,
        extra_instructions: Optional[str] = None,
        font_name: Optional[str] = None,
        background_info: Optional[str] = None,
        color_schema: Optional[str] = None,
        image_provider: Optional[str] = None,
        image_model: Optional[str] = None,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Process an article URL and generate social media content (carousel post).

        Args:
            url: Article URL to process
            max_slides: Maximum number of slides to generate (default: 10)
            username: Social media username (e.g., "@robots")
            tagline: Tagline/brand message (e.g., "daily programming tips & tricks")
            title: Custom title to override scraped article title
            extra_instructions: Additional instructions for the LLM
            font_name: Font name for slides
            background_info: Background description
            color_schema: Color schema description
            image_provider: Image generation provider ("openrouter" or "openai")
            image_model: Image generation model (e.g., "dall-e-3", "google/gemini-2.5-flash-image")
            thread_id: Thread ID for conversation tracking

        Returns:
            Dictionary containing carousel slides with images
        """
        try:
            if not url:
                raise ValueError("URL must be provided")

            initial_state = {
                "messages": [],
                "url": url,
                "max_slides": max_slides,
                "username": username,
                "tagline": tagline,
                "title": title,
                "extra_instructions": extra_instructions,
                "font_name": font_name,
                "background_info": background_info,
                "color_schema": color_schema,
                "image_provider": image_provider,
                "image_model": image_model,
                "output_folder": None,
                "article_content": None,
                "slides": None,
                "slides_with_images": None,
                "status": "initialized",
                "error": None,
            }

            print("\n" + "=" * 80)
            print("🚀 Starting Social Media Content Generator Agent Workflow")
            print("=" * 80)
            config = {"configurable": {"thread_id": thread_id}}
            result = None

            async for event in self.graph.astream(initial_state, config):
                result = event
                if "agent" in event:
                    status = event["agent"].get("status", "processing")
                    logger.info(f"Agent status: {status}")

            print("\n" + "=" * 80)
            print("✅ Workflow completed successfully!")
            print("=" * 80)

            final_state = (
                result.get("agent", initial_state) if result else initial_state
            )

            if final_state.get("status") == "error":
                error_msg = final_state.get("error", "Unknown error occurred")
                raise ValueError(f"Agent processing failed: {error_msg}")

            return {
                "status": "success",
                "url": url,
                "article_title": final_state.get("article_content", {}).get(
                    "title", "Unknown"
                ),
                "total_slides": len(final_state.get("slides_with_images", [])),
                "slides": final_state.get("slides_with_images", []),
                "processing_status": final_state.get("status", "unknown"),
            }

        except Exception as e:
            logger.error(f"Error processing social media content request: {str(e)}")
            raise


# ============================================================================
# Factory Function
# ============================================================================


def create_agent() -> SocialMediaContentGeneratorAgent:
    """Factory function to create a new social media content generator agent instance."""
    return SocialMediaContentGeneratorAgent()


# ============================================================================
# Utility Functions
# ============================================================================


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Sanitize text to be used as a filename.

    Args:
        text: The text to sanitize
        max_length: Maximum length of the filename

    Returns:
        Sanitized filename-safe string
    """
    invalid_chars = '<>:"/\\|?*.'
    sanitized = text.strip()

    for char in invalid_chars:
        sanitized = sanitized.replace(char, "-")

    sanitized = re.sub(r"[\s\-]+", "-", sanitized)
    sanitized = sanitized.strip("-")

    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip("-")

    return sanitized if sanitized else "social_media_content"


# ============================================================================
# Single Image Generation Helper
# ============================================================================


async def generate_single_informational_image(
    url: str,
    username: Optional[str] = None,
    tagline: Optional[str] = None,
    title: Optional[str] = None,
    extra_instructions: Optional[str] = None,
    font_name: Optional[str] = None,
    background_info: Optional[str] = None,
    color_schema: Optional[str] = None,
    image_provider: Optional[str] = None,
    image_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a single informational image from an article URL.

    Args:
        url: Article URL to process
        username: Social media username (e.g., "@robots")
        tagline: Tagline/brand message (e.g., "daily programming tips & tricks")
        title: Custom title to override scraped article title
        extra_instructions: Additional instructions for the LLM

    Returns:
        Dictionary containing image information and metadata
    """
    try:
        # Scrape article content
        logger.info(f"Scraping article from: {url}")
        article_content = await scrape_article_content(url)

        if not article_content or not article_content.get("fullText"):
            raise ValueError("No article content found")

        # Create output folder based on article title
        article_title = article_content.get("title", "informational_image")
        sanitized_title = sanitize_filename(article_title)

        try:
            script_dir = Path(__file__).parent.absolute()
            base_output_dir = script_dir / "output"
        except NameError:
            base_output_dir = Path.cwd() / "output"

        # Create a folder for this image
        image_folder = base_output_dir / sanitized_title
        image_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output folder: {image_folder}")

        # Generate content using LLM
        llm_service = LLMService()
        image_content = llm_service.generate_single_informational_image(
            article_content,
            username=username,
            tagline=tagline,
            title=title,
            extra_instructions=extra_instructions,
            font_name=font_name,
            background_info=background_info,
            color_schema=color_schema,
        )

        # Generate the image
        image_prompt = image_content.get("image_prompt", "")
        logger.info("Generating informational image...")

        image_provider_param = image_provider or IMAGE_PROVIDER
        image_model_param = image_model or IMAGE_MODEL
        image_info = generate_carousel_image(
            image_prompt,
            1,
            image_folder,
            orientation="square",
            provider=image_provider_param,
            model=image_model_param,
        )

        if not image_info:
            raise ValueError("Failed to generate image")

        return {
            "status": "success",
            "url": url,
            "article_title": article_title,
            "is_list_post": image_content.get("is_list_post", False),
            "title": image_content.get("title", ""),
            "content": image_content.get("content", ""),
            "image_prompt": image_prompt,
            "image_path": image_info["path"],
            "image_filename": image_info["filename"],
            "image_relative_path": image_info["relative_path"],
        }

    except Exception as e:
        logger.error(f"Error generating single informational image: {str(e)}")
        raise


async def generate_infographic_from_prompt(
    user_prompt: str,
    username: Optional[str] = None,
    tagline: Optional[str] = None,
    font_name: Optional[str] = None,
    background_info: Optional[str] = None,
    color_schema: Optional[str] = None,
    extra_instructions: Optional[str] = None,
    image_provider: Optional[str] = None,
    image_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an infographic from a user text prompt.

    Args:
        user_prompt: User's text prompt (e.g., "top 10 free AI tools")
        username: Social media username (e.g., "@robots")
        tagline: Tagline/brand message (e.g., "daily programming tips & tricks")
        font_name: Font name for the infographic
        background_info: Background description
        color_schema: Color schema description
        extra_instructions: Additional instructions for the LLM

    Returns:
        Dictionary containing infographic information and metadata
    """
    try:
        # Create output folder based on prompt
        sanitized_prompt = sanitize_filename(user_prompt[:50])

        try:
            script_dir = Path(__file__).parent.absolute()
            base_output_dir = script_dir / "output"
        except NameError:
            base_output_dir = Path.cwd() / "output"

        # Create a folder for this infographic
        infographic_folder = base_output_dir / sanitized_prompt
        infographic_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output folder: {infographic_folder}")

        # Generate content using LLM
        llm_service = LLMService()
        infographic_content = llm_service.generate_infographic_from_prompt(
            user_prompt=user_prompt,
            username=username,
            tagline=tagline,
            font_name=font_name,
            background_info=background_info,
            color_schema=color_schema,
            extra_instructions=extra_instructions,
        )

        # Generate the image
        image_prompt = infographic_content.get("image_prompt", "")
        infographic_type = infographic_content.get("type", "unknown")
        logger.info(f"Generating infographic image (type: {infographic_type})...")

        image_provider_param = image_provider or IMAGE_PROVIDER
        image_model_param = image_model or IMAGE_MODEL
        image_info = generate_carousel_image(
            image_prompt,
            1,
            infographic_folder,
            orientation="square",
            provider=image_provider_param,
            model=image_model_param,
        )

        if not image_info:
            raise ValueError("Failed to generate image")

        return {
            "status": "success",
            "prompt": user_prompt,
            "type": infographic_type,
            "title": infographic_content.get("title", ""),
            "content": infographic_content.get("content", ""),
            "image_prompt": image_prompt,
            "image_path": image_info["path"],
            "image_filename": image_info["filename"],
            "image_relative_path": image_info["relative_path"],
        }

    except Exception as e:
        logger.error(f"Error generating infographic from prompt: {str(e)}")
        raise


async def generate_carousel_from_prompt(
    user_prompt: str,
    max_slides: int = 10,
    username: Optional[str] = None,
    tagline: Optional[str] = None,
    font_name: Optional[str] = None,
    background_info: Optional[str] = None,
    color_schema: Optional[str] = None,
    extra_instructions: Optional[str] = None,
    image_provider: Optional[str] = None,
    image_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a carousel post from a user text prompt.

    Args:
        user_prompt: User's text prompt (e.g., "top 10 free AI tools", "how to learn React")
        max_slides: Maximum number of slides to generate
        username: Social media username (e.g., "@robots")
        tagline: Tagline/brand message (e.g., "daily programming tips & tricks")
        font_name: Font name for the carousel slides
        background_info: Background description
        color_schema: Color schema description
        extra_instructions: Additional instructions for the LLM
        image_provider: Image generation provider ("openrouter" or "openai")
        image_model: Image generation model (e.g., "dall-e-3", "google/gemini-2.5-flash-image")

    Returns:
        Dictionary containing carousel slides with images
    """
    try:
        # Create output folder based on prompt
        sanitized_prompt = sanitize_filename(user_prompt[:50])

        try:
            script_dir = Path(__file__).parent.absolute()
            base_output_dir = script_dir / "output"
        except NameError:
            base_output_dir = Path.cwd() / "output"

        # Create a folder for this carousel
        carousel_folder = base_output_dir / sanitized_prompt
        carousel_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output folder: {carousel_folder}")

        # Generate content using LLM
        llm_service = LLMService()
        carousel_slides = llm_service.generate_carousel_from_prompt(
            user_prompt=user_prompt,
            max_slides=max_slides,
            username=username,
            tagline=tagline,
            font_name=font_name,
            background_info=background_info,
            color_schema=color_schema,
            extra_instructions=extra_instructions,
        )

        # Generate images for each slide
        image_provider_param = image_provider or IMAGE_PROVIDER
        image_model_param = image_model or IMAGE_MODEL
        slides_with_images = []

        for slide in carousel_slides:
            slide_number = slide.get("slide_number", 0)
            image_prompt = slide.get("image_prompt", "")
            title = slide.get("title", "")

            logger.info(f"Generating image for slide {slide_number}: {title[:50]}...")

            image_info = generate_carousel_image(
                image_prompt,
                slide_number,
                carousel_folder,
                orientation="square",
                provider=image_provider_param,
                model=image_model_param,
            )

            if not image_info:
                logger.warning(f"Failed to generate image for slide {slide_number}")
                continue

            slide["image_path"] = image_info["path"]
            slide["image_filename"] = image_info["filename"]
            slide["image_relative_path"] = image_info["relative_path"]
            slides_with_images.append(slide)

        if not slides_with_images:
            raise ValueError("Failed to generate any slides with images")

        # Determine a title from the first slide or prompt
        carousel_title = (
            slides_with_images[0].get("title", sanitized_prompt)
            if slides_with_images
            else sanitized_prompt
        )

        return {
            "status": "success",
            "prompt": user_prompt,
            "article_title": carousel_title,
            "total_slides": len(slides_with_images),
            "slides": slides_with_images,
            "processing_status": "completed",
        }

    except Exception as e:
        logger.error(f"Error generating carousel from prompt: {str(e)}")
        raise


async def generate_infographic_with_reference_image(
    user_prompt: str,
    reference_image_bytes: bytes,
    username: Optional[str] = None,
    tagline: Optional[str] = None,
    font_name: Optional[str] = None,
    background_info: Optional[str] = None,
    color_schema: Optional[str] = None,
    image_provider: Optional[str] = None,
    image_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an infographic from a user text prompt using a reference image as design guide.

    Args:
        user_prompt: User's text prompt describing what infographic to create
        reference_image_bytes: Reference image file bytes to use as design guide
        username: Social media username (e.g., "@robots")
        tagline: Tagline/brand message (e.g., "daily programming tips & tricks")
        font_name: Font name for the infographic
        background_info: Background description
        color_schema: Color schema description

    Returns:
        Dictionary containing infographic information and metadata
    """
    try:
        # Convert image to base64
        reference_image_base64 = base64.b64encode(reference_image_bytes).decode("utf-8")

        # Create output folder based on prompt
        sanitized_prompt = sanitize_filename(user_prompt[:50])

        try:
            script_dir = Path(__file__).parent.absolute()
            base_output_dir = script_dir / "output"
        except NameError:
            base_output_dir = Path.cwd() / "output"

        # Create a folder for this infographic
        infographic_folder = base_output_dir / f"{sanitized_prompt}_with_reference"
        infographic_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output folder: {infographic_folder}")

        # Generate content using LLM (same as generate_infographic_from_prompt)
        llm_service = LLMService()
        infographic_content = llm_service.generate_infographic_from_prompt(
            user_prompt=user_prompt,
            username=username,
            tagline=tagline,
            font_name=font_name,
            background_info=background_info,
            color_schema=color_schema,
        )

        # Extract content from LLM response for the reference-based generation
        title = infographic_content.get("title", "")
        content = infographic_content.get("content", "")

        # Create a more focused content-only prompt
        enhanced_prompt = f"""TITLE: {title}

CONTENT:
{content}

NOTE: The reference image will define all design elements. Just ensure the above text content is properly placed in the new design following the reference image's exact layout and typography."""

        # Generate the image with reference
        infographic_type = infographic_content.get("type", "unknown")
        logger.info(
            f"Generating infographic image with reference (type: {infographic_type})..."
        )

        image_provider_param = image_provider or IMAGE_PROVIDER
        image_model_param = image_model or IMAGE_MODEL
        image_info = generate_carousel_image(
            enhanced_prompt,
            1,
            infographic_folder,
            orientation="square",
            reference_image_base64=reference_image_base64,
            provider=image_provider_param,
            model=image_model_param,
        )

        if not image_info:
            raise ValueError("Failed to generate image")

        return {
            "status": "success",
            "prompt": user_prompt,
            "type": infographic_type,
            "title": infographic_content.get("title", ""),
            "content": infographic_content.get("content", ""),
            "image_prompt": enhanced_prompt,
            "image_path": image_info["path"],
            "image_filename": image_info["filename"],
            "image_relative_path": image_info["relative_path"],
        }

    except Exception as e:
        logger.error(f"Error generating infographic with reference image: {str(e)}")
        raise

"""
Standalone LangGraph Agent for Carousel Post Generation

This is a production-ready, standalone backend agent that uses LangGraph
to orchestrate carousel post generation from article content.

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


# ============================================================================
# State Definition
# ============================================================================


class CarouselPostState(TypedDict):
    """State for the carousel post generator agent."""

    messages: Annotated[List, lambda x, y: x + y]
    url: str
    max_slides: int
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
# Image Generation (OpenRouter Gemini)
# ============================================================================


def generate_carousel_image(
    prompt: str, slide_number: int, output_folder: Path, orientation: str = "square"
) -> Optional[Dict[str, str]]:
    """
    Generate an image for a carousel slide using OpenRouter Gemini API and save locally.

    Args:
        prompt: Image generation prompt
        slide_number: Slide number for naming
        output_folder: Folder to save the image in
        orientation: Image orientation (square recommended for carousels)

    Returns:
        Dictionary with local image path and metadata, or None if failed
    """
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not found in environment variables")
        return None

    try:
        # Create carousel-optimized prompt
        full_prompt = f"Create a {orientation} carousel slide image: {prompt}. Make it clean, modern, and visually engaging with clear focal point. Perfect for social media carousel."

        logger.info(f"Generating image for slide {slide_number}: {prompt}")

        # OpenRouter API endpoint
        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://contentrob.com",
            "X-Title": "ContentRob",
        }

        payload = {
            "model": "google/gemini-2.5-flash-image",
            "messages": [{"role": "user", "content": full_prompt}],
            "modalities": ["image", "text"],
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code == 200:
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                message = choice.get("message", {})

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

    def __init__(self, model_name: str = "gpt-5", temperature: float = 1):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
        )

    def generate_carousel_slides(
        self, article_content: Dict[str, Any], max_slides: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate carousel slide content from article.

        Args:
            article_content: Scraped article content
            max_slides: Maximum number of slides to generate

        Returns:
            List of slide dictionaries with title, content, and image prompt
        """
        try:
            title = article_content.get("title", "Unknown")
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
  - Prominent text: "Follow @robots for daily programming tips & tricks →"
  - Optional: your logo or handle in the corner

=== DESIGN RULES FOR EVERY IMAGE_PROMPT (CRITICAL) ===
All slides must look like professional Canva-style carousel slides:
- Format: 1080x1080px or 1080x1350px (square or vertical)
- Background: Clean gradient or subtle tech/coding theme (dark navy/blue/purple or modern light mode)
- Color scheme: Consistent across all slides (e.g., navy background, white + cyan accent text)
- Title: Extra large bold sans-serif font (e.g., Montserrat Black), top 20% of slide
- Body text: Clean bullet points, highly readable, max 7 lines
- Add "Slide X of {max_slides}" in small text at top-right or bottom-right
- Add subtle relevant icons (code symbols, laptop, lightbulb, rocket, etc.)
- Add small "@robots" handle in bottom-left or bottom-right corner on every slide
- High contrast, modern, premium feel — looks expensive

=== OUTPUT FORMAT ===
Return ONLY a valid JSON array (no markdown, no explanation). Each object must have exactly these keys:

[
  {{
    "slide_number": 1,
    "title": "Exact title text that will appear on the image",
    "content": "Exact body text that will appear on the image (use \\n for line breaks in bullets)",
    "image_prompt": "Extremely detailed prompt that forces Gemini to render the exact title and content as text on the image. Include layout, colors, fonts, and all text verbatim."
  }}
]


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


# ============================================================================
# Graph Nodes
# ============================================================================


async def scrape_article_node(state: CarouselPostState) -> CarouselPostState:
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

        # Create a folder for this carousel post
        carousel_folder = base_output_dir / sanitized_title
        carousel_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output folder: {carousel_folder}")

        print(f"✅ Successfully scraped article: {article_content.get('title')}")
        return {
            **state,
            "article_content": article_content,
            "output_folder": carousel_folder,
            "status": "scraped",
        }
    except Exception as e:
        print(f"❌ Error in scrape_article_node: {str(e)}")
        logger.error(f"Error in scrape_article_node: {str(e)}")
        return {**state, "status": "error", "error": str(e)}


async def generate_slides_node(state: CarouselPostState) -> CarouselPostState:
    """Generate carousel slide content using LLM."""
    try:
        print("\n" + "=" * 80)
        print("📝 NODE: generate_slides_node - Generating carousel slides")
        print("=" * 80)
        logger.info("Generating carousel slides")

        llm_service = LLMService()
        slides = llm_service.generate_carousel_slides(
            state["article_content"], state.get("max_slides", 10)
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


async def generate_images_node(state: CarouselPostState) -> CarouselPostState:
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
            image_info = generate_carousel_image(
                image_prompt, slide_number, output_folder
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


async def agent_node(state: CarouselPostState) -> CarouselPostState:
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


def should_continue(state: CarouselPostState) -> Literal["continue", "end"]:
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


def create_carousel_post_generator_agent():
    """Create and compile the carousel post generator LangGraph agent."""

    workflow = StateGraph(CarouselPostState)

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


class CarouselPostGeneratorAgent:
    """Standalone LangGraph agent for carousel post generation."""

    def __init__(self):
        self.graph = create_carousel_post_generator_agent()
        logger.info("Carousel Post Generator Agent initialized")

    async def process(
        self,
        url: str,
        max_slides: int = 10,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Process an article URL and generate carousel post.

        Args:
            url: Article URL to process
            max_slides: Maximum number of slides to generate (default: 10)
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
                "output_folder": None,
                "article_content": None,
                "slides": None,
                "slides_with_images": None,
                "status": "initialized",
                "error": None,
            }

            print("\n" + "=" * 80)
            print("🚀 Starting Carousel Post Generator Agent Workflow")
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
            logger.error(f"Error processing carousel post request: {str(e)}")
            raise


# ============================================================================
# Factory Function
# ============================================================================


def create_agent() -> CarouselPostGeneratorAgent:
    """Factory function to create a new carousel post generator agent instance."""
    return CarouselPostGeneratorAgent()


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

    return sanitized if sanitized else "carousel_post"

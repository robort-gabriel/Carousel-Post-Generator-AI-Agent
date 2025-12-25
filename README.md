# Social Media Content Generator 🎨

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Social Media Content Generator** is a production-ready FastAPI application powered by LangGraph that automatically generates engaging social media content. This AI agent can create:
- **Carousel Posts**: Multi-slide carousel posts with custom images for each slide
- **Single Informational Images**: One comprehensive image for list posts or article summaries
- **Infographics from Prompts**: Generate infographics directly from text prompts (e.g., "top 10 AI tools")
- **Infographics with Reference Images**: Create new infographics matching an existing design style

The agent uses advanced language models to create content and generates custom images using **Google Gemini via OpenRouter** or **OpenAI DALL-E/GPT-Image models**.

## 📑 Table of Contents

- [Features](#-features)
- [Security Features](#-security-features)
- [Architecture](#️-architecture)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [API Usage](#-api-usage)
- [Output](#-output)
- [Use Cases](#-use-cases)
- [Best Practices](#-best-practices)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Production Deployment](#-production-deployment)
- [How It Works](#-how-it-works)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Features

- **Automatic Article Scraping**: Uses Playwright to scrape article content from any URL
- **Multiple Content Types**: 
  - Carousel posts from article URLs
  - Single informational images from article URLs
  - Infographics from text prompts (no URL needed)
  - Infographics with reference image design matching
- **Smart Content Detection**: Automatically detects list posts vs. general articles for optimal image generation
- **AI-Powered Content Generation**: Uses GPT models to create engaging social media content
- **Multiple Image Providers**: Supports both OpenRouter (Gemini) and OpenAI (DALL-E, GPT-Image) for image generation
- **Custom Image Generation**: Generates unique images using Google Gemini via OpenRouter or OpenAI models
- **Design Customization**: Customize fonts, backgrounds, and color schemes for all content types
- **Reference Image Support**: Upload a reference image to match its exact design style (GPT-Image models)
- **Frontend-Ready API**: Image serving endpoint, request ID tracking, consistent JSON responses
- **Image URLs**: All responses include direct URLs to access generated images via API
- **Local Image Storage**: Automatically saves generated images in organized folders
- **Structured Output**: Saves content in both Markdown and JSON formats with images in the same folder
- **Configurable Slides**: Customize the number of slides for carousels (3-15, default: 10)
- **Custom Branding**: Set username and tagline for consistent branding across all content
- **LangGraph Workflow**: Production-ready agent following best practices
- **RESTful API**: FastAPI endpoints with security features

## 🔐 Security Features

- **API Key Authentication** - Secure header-based authentication
- **Rate Limiting** - Configurable per-minute rate limits
- **Input Validation** - Pydantic models for request validation
- **CORS Support** - Configurable cross-origin resource sharing
- **Error Handling** - Comprehensive error handling with proper HTTP status codes

## 🏗️ Architecture

**Social Media Content Generator** is built using **LangGraph** and **FastAPI** with:
- **State Management**: TypedDict-based state tracking
- **Web Scraping**: Playwright for article content extraction
- **LLM Orchestration**: GPT models for content generation
- **Image Generation**: Multiple providers (OpenRouter/Gemini, OpenAI DALL-E, OpenAI GPT-Image)
- **Local Storage**: Automatic file organization in output folders
- **Image Serving**: API endpoint to serve generated images directly
- **Error Handling**: Global exception handlers with consistent JSON error responses
- **Request Tracking**: Request ID middleware for debugging and logging

### Workflow

**For Carousel Posts:**
```
Article URL → Scrape Content → Generate Slides → Generate Images → Save Files → API Response
```

**For Single Images:**
```
Article URL → Scrape Content → Analyze (List/General) → Generate Content → Generate Image → Save File → API Response
```

**For Infographics from Prompts:**
```
Text Prompt → Analyze Type (List/Summary) → Generate Content → Generate Image → Save File → API Response
```

**For Infographics with Reference Images:**
```
Text Prompt + Reference Image → Generate Content → Match Reference Design → Generate Image → Save File → API Response
```

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.11+** installed on your system
- **OpenAI API key** (required for content generation and optional for image generation)
- **OpenRouter API key** (required for image generation if using OpenRouter provider)
- **API key for authentication** (optional, set `API_KEY` in `.env` for production)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/carousel-post-generator.git
cd carousel-post-generator
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional - Image Generation Provider
# Choose one: "openrouter" (requires OPENROUTER_API_KEY) or "openai" (uses OpenAI DALL-E/GPT-Image)
IMAGE_PROVIDER=openrouter
IMAGE_MODEL=google/gemini-2.5-flash-image  # or dall-e-3, gpt-image-1.5, etc.

# Optional - OpenRouter (required if IMAGE_PROVIDER=openrouter)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional - Security
API_KEY=your_api_key_for_authentication
RATE_LIMIT_PER_MINUTE=10

# Optional - Server Configuration
PORT=8000
HOST=0.0.0.0
CORS_ORIGINS=http://localhost:3000,http://localhost:5173  # Frontend URLs
DEBUG=False
```

### 5. Start the API Server

```bash
./start_api.sh
```

Or manually:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📚 API Usage

The **Social Media Content Generator** provides a RESTful API for generating social media content.

### Health Check

```bash
curl http://localhost:8000/health
```

### Get API Information

```bash
curl http://localhost:8000/api/v1/info
```

### Generate Carousel Post

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "max_slides": 10,
    "username": "@yourhandle",
    "tagline": "your brand tagline",
    "font_name": "Inter",
    "background_info": "Dark background #101010",
    "color_schema": "White text with cyan accents",
    "image_provider": "openai",
    "image_model": "dall-e-3",
    "extra_instructions": "Use a minimalist design style"
  }'
```

### Generate Single Informational Image

```bash
curl -X POST "http://localhost:8000/api/v1/generate-single-image" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "username": "@yourhandle",
    "tagline": "your brand tagline",
    "font_name": "Roboto",
    "background_info": "Light gradient background",
    "color_schema": "Dark text on light background",
    "extra_instructions": "Focus on beginner-friendly language"
  }'
```

### Generate Infographic from Text Prompt

```bash
curl -X POST "http://localhost:8000/api/v1/generate-infographic-from-prompt" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "top 10 free AI tools",
    "username": "@yourhandle",
    "tagline": "daily programming tips",
    "font_name": "Inter",
    "background_info": "Solid dark background using #101010",
    "color_schema": "#101010 background with white text and cyan accents"
  }'
```

### Generate Infographic with Reference Image

```bash
curl -X POST "http://localhost:8000/api/v1/generate-infographic-with-reference" \
  -H "X-API-Key: your_api_key_here" \
  -F "reference_image=@/path/to/reference-image.png" \
  -F "prompt=top 10 free AI tools" \
  -F "username=@yourhandle" \
  -F "tagline=daily programming tips" \
  -F "font_name=Inter" \
  -F "background_info=Dark background #101010" \
  -F "color_schema=White text with cyan accents" \
  -F "image_provider=openai" \
  -F "image_model=gpt-image-1.5"
```

### Access Generated Images

```bash
# Images are automatically served via API
curl http://localhost:8000/api/v1/images/slide_1_image.png
```

### Available API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/api/v1/info` | GET | API information and capabilities |
| `/api/v1/generate` | POST | Generate carousel post from article URL |
| `/api/v1/generate-carousel-from-prompt` | POST | Generate carousel post from text prompt |
| `/api/v1/generate-single-image` | POST | Generate single informational image from article URL |
| `/api/v1/generate-infographic-from-prompt` | POST | Generate infographic from text prompt |
| `/api/v1/generate-infographic-with-reference` | POST | Generate infographic with reference image design matching |
| `/api/v1/images/{filename}` | GET | Serve generated images via API |

### Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example Carousel Request

```json
{
  "url": "https://example.com/article",
  "max_slides": 8,
  "username": "@yourhandle",
  "tagline": "daily tips & tricks",
  "font_name": "Inter",
  "background_info": "Dark navy gradient background",
  "color_schema": "Navy background with white and cyan accent text",
  "extra_instructions": "Use a modern, clean design"
}
```

### Example Carousel Response

```json
{
  "success": true,
  "status": "success",
  "url": "https://example.com/article",
  "article_title": "How to Create Engaging Content",
  "total_slides": 8,
  "slides": [
    {
      "slide_number": 1,
      "title": "Hook Title Here",
      "content": "Attention-grabbing opening content.",
      "image_prompt": "Detailed prompt for image generation",
      "image_path": "/full/path/to/slide_1_hook_title.png",
      "image_url": "http://localhost:8000/api/v1/images/slide_1_hook_title.png",
      "image_filename": "slide_1_hook_title.png",
      "image_relative_path": "./slide_1_hook_title.png"
    }
  ],
  "processing_status": "completed",
  "request_id": "abc-123-def-456"
}
```

### Example Single Image Request

```json
{
  "url": "https://example.com/list-article",
  "username": "@yourhandle",
  "tagline": "daily tips & tricks",
  "font_name": "Roboto",
  "background_info": "Clean gradient background",
  "color_schema": "Modern and visually appealing color scheme"
}
```

### Example Infographic from Prompt Request

```json
{
  "prompt": "top 10 free AI tools",
  "username": "@yourhandle",
  "tagline": "daily programming tips",
  "font_name": "Inter",
  "background_info": "Solid dark background using #101010",
  "color_schema": "#101010 background with white text and cyan accents"
}
```

### Example Infographic with Reference Image Request

**Note**: This endpoint uses `multipart/form-data` format:

```bash
# Using curl
curl -X POST "http://localhost:8000/api/v1/generate-infographic-with-reference" \
  -H "X-API-Key: your_api_key_here" \
  -F "reference_image=@reference.png" \
  -F "prompt=top 10 free AI tools" \
  -F "username=@yourhandle" \
  -F "tagline=daily tips" \
  -F "font_name=Inter"
```

### Example Single Image Response

```json
{
  "success": true,
  "status": "success",
  "url": "https://example.com/list-article",
  "article_title": "Top 10 Programming Tips",
  "title": "Top 10 Programming Tips",
  "image_path": "/full/path/to/slide_1_top_10_tips.png",
  "image_url": "http://localhost:8000/api/v1/images/slide_1_top_10_tips.png",
  "request_id": "abc-123-def-456"
}
```

### Example Error Response

```json
{
  "success": false,
  "error": "Validation Error",
  "detail": "Invalid request data",
  "errors": [
    {
      "field": "url",
      "message": "field required",
      "type": "value_error.missing"
    }
  ],
  "error_type": "validation_error",
  "request_id": "abc-123-def-456"
}
```

## 📊 Output

The **Social Media Content Generator** automatically saves results to the `output/` directory:

### File Structure

All files are saved in a folder named after the article title: `output/{article_title}/`

**For Carousel Posts:**
- **`carousel_{timestamp}.md`** - Human-readable markdown format with slide content and images
- **`carousel_{timestamp}.json`** - Machine-readable JSON format for further processing
- **`slide_1_*.png`** - Generated images for each slide
- **`slide_2_*.png`** - ...
- **`slide_N_*.png`** - Last slide image

**For Single Images:**
- **`slide_1_*.png`** - Single generated informational image

**For Infographics from Prompts:**
- **`slide_1_*.png`** - Generated infographic image
- Files saved in folder named after the prompt

**For Infographics with Reference Images:**
- **`slide_1_*.png`** - Generated infographic matching reference design
- Files saved in folder named after the prompt with `_with_reference` suffix

Example structure:
```
output/
└── coding-programs-for-beginners/
    ├── carousel_20241201_103000.md
    ├── carousel_20241201_103000.json
    ├── slide_1_hook_title.png
    ├── slide_2_key_concept.png
    ├── slide_3_important_tip.png
    └── ...
```

## 🎯 Use Cases

Perfect for:

- **Social Media Content**: Create engaging Instagram/LinkedIn carousels and single images that drive engagement
- **Content Repurposing**: Convert blog posts into shareable formats for multiple platforms
- **List Posts**: Automatically detect and create beautiful list-based informational images
- **Article Summaries**: Generate single comprehensive images summarizing key article points
- **Prompt-Based Infographics**: Create infographics directly from text prompts without needing article URLs
- **Design Consistency**: Use reference images to maintain consistent design style across multiple infographics
- **Educational Content**: Break down complex topics into digestible, visual content
- **Marketing**: Transform product articles into promotional content with custom visuals
- **Personal Branding**: Create thought leadership content from your articles with consistent branding
- **Content Automation**: Automate social media content creation workflow for content teams
- **Template-Based Generation**: Upload a reference design and generate new content matching that style

## 📈 Best Practices

### Slide Count

- **8-10 slides**: Optimal for most articles (recommended)
- **6-8 slides**: Good for shorter articles or quick tips
- **10-12 slides**: Suitable for comprehensive guides or tutorials
- **Avoid >12 slides**: May lose audience engagement

### Article Selection

Best results with:
- How-to guides and tutorials
- Listicles (Top 10, Best practices, etc.)
- Educational content with clear structure
- Articles with distinct sections or points

### Image Generation

- Each image takes ~5-10 seconds to generate
- Images are automatically saved as PNG files in the content folder
- Optimized prompts for consistent style across all content
- Square orientation recommended for social media
- Images are saved with descriptive filenames based on their prompts
- Single images automatically detect list posts vs. general articles
- Reference images can be used to match exact design styles
- Custom design parameters (font, background, colors) available for all endpoints

### Design Customization

All endpoints support custom design parameters:

- **`font_name`**: Specify font family (e.g., "Inter", "Roboto", "Montserrat")
- **`background_info`**: Describe background style (e.g., "Dark background #101010", "Light gradient")
- **`color_schema`**: Define color scheme (e.g., "White text with cyan accents", "Navy background with white text")

These parameters have sensible defaults if not provided.

### Image Generation Providers

You can choose between two image generation providers:

**OpenRouter (Default):**
- Models: `google/gemini-2.5-flash-image`, `google/gemini-3-pro-image-preview`, `x-ai/grok-4.1-fast`
- Supports reference images via vision models
- Requires `OPENROUTER_API_KEY`

**OpenAI:**
- Models: `dall-e-3`, `dall-e-2`, `gpt-image-1`, `gpt-image-1.5`
- GPT-Image models support reference images via `/v1/images/edits` endpoint
- DALL-E models don't support reference images
- Uses `OPENAI_API_KEY`

Set via request parameter or environment variable:
```env
IMAGE_PROVIDER=openai
IMAGE_MODEL=gpt-image-1.5
```

## 🔧 Configuration

### Rate Limiting

Adjust rate limits in `.env`:

```env
RATE_LIMIT_PER_MINUTE=10  # Requests per minute per IP
```

### CORS Configuration

Configure allowed origins:

```env
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### API Key Authentication

Set `API_KEY` in `.env` to enable authentication. If not set, API is accessible without authentication (not recommended for production).

### Image Provider Configuration

Configure which image generation provider to use:

```env
# Use OpenRouter (default)
IMAGE_PROVIDER=openrouter
IMAGE_MODEL=google/gemini-2.5-flash-image

# Or use OpenAI
IMAGE_PROVIDER=openai
IMAGE_MODEL=dall-e-3  # or gpt-image-1.5 for reference image support
```

## 🐛 Troubleshooting

### Issue: API Key Not Working

**Solution:**
1. Check that `API_KEY` is set in `.env`
2. Ensure the `X-API-Key` header is included in requests
3. Verify the API key matches exactly (no extra spaces)

### Issue: Rate Limit Exceeded

**Solution:**
1. Wait for the rate limit window to reset (1 minute)
2. Increase `RATE_LIMIT_PER_MINUTE` in `.env`
3. Use different IP addresses for higher throughput

### Issue: Playwright Installation Issues

If `playwright install chromium` fails:

```bash
# Try installing all browsers
playwright install

# Or use system browser
playwright install --with-deps chromium
```

### Issue: Image Generation Fails

- Check `OPENROUTER_API_KEY` is valid
- Check OpenRouter API status
- Ensure you have OpenRouter credits
- Verify you have write permissions for the output folder

### Issue: Article Scraping Fails

- Verify the URL is accessible
- Check if the website blocks automated access
- Ensure Playwright is properly installed
- Review API logs for specific error messages

### Issue: Local File Save Fails

- Ensure the output directory is writable
- Check available disk space
- Verify file path permissions

## 📝 Content Structure

### Carousel Slide Structure

Each carousel post includes:

1. **Slide 1 (Hook)**: Attention-grabbing title slide that introduces the topic
2. **Slides 2-N-1 (Content)**: Key points and insights from the article
3. **Slide N (CTA)**: Call-to-action or conclusion slide with username and tagline

### Slide Components

Each slide contains:
- **Slide Number**: Sequential number (1-N)
- **Title**: Catchy title (max 60 characters)
- **Content**: Concise, valuable content (2-3 sentences, max 150 characters)
- **Image**: Custom-generated image relevant to the slide content
- **Image Prompt**: The prompt used to generate the image

### Single Image Structure

Single informational images include:

- **List Posts**: Automatically extracts all list items and displays them in one organized image
- **General Articles**: Summarizes key takeaways (3-5 main points) with a brief summary
- **Custom Branding**: Includes username and tagline on every image
- **Smart Detection**: Automatically determines if content is a list post or general article

## 🚀 Production Deployment

### Using Gunicorn

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium

COPY . .

CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

## 📁 Project Structure

```
carousel-post-generator/
├── main.py                              # FastAPI application for Social Media Content Generator
├── carousel_post_generator_agent.py     # LangGraph agent implementation
├── output/                              # Generated content and images
├── requirements.txt                     # Python dependencies
├── start_api.sh                         # Startup script
└── README.md                            # This file
```

## 🔄 How It Works

### 1. Article Scraping (Playwright)

The **Social Media Content Generator** uses Playwright to:
- Navigate to the article URL
- Wait for dynamic content to load
- Extract article title, content, and metadata
- Parse headings and paragraphs
- Clean and structure the content

### 2. Content Generation (GPT Models)

**For Carousel Posts:**
The AI agent's LLM:
- Analyzes the article content
- Identifies key points and insights
- Creates engaging, concise slide content
- Generates detailed image prompts for each slide
- Ensures slides tell a cohesive story

**For Single Images:**
The AI agent's LLM:
- Analyzes if the article is a list post or general article
- For list posts: Extracts all list items
- For general articles: Summarizes key takeaways
- Generates detailed image prompt with all content
- Includes username and tagline in the design

**For Infographics from Prompts:**
The AI agent's LLM:
- Analyzes the text prompt to determine if it's a list type or summary type
- For list types (e.g., "top 10 tools"): Generates the list items based on the prompt
- For summary types (e.g., "how to use React"): Creates summary with key points
- Generates detailed image prompt with the content
- Applies custom design parameters if provided

**For Infographics with Reference Images:**
The AI agent:
- Uses the reference image as an exact design template
- Generates content based on the text prompt
- Instructs Gemini to replicate the reference image's design exactly
- Matches colors, layout, typography, and visual elements from the reference
- Only changes the text content while maintaining identical design style

### 3. Image Generation (Multiple Providers)

For each image, the **Social Media Content Generator** supports two providers:

**OpenRouter (Gemini):**
- Uses the LLM-generated image prompt
- For reference-based generation: Includes the reference image in the API call
- Calls OpenRouter's Google Gemini image generation API
- Receives base64-encoded image
- Decodes and saves as PNG file locally

**OpenAI (DALL-E or GPT-Image):**
- Uses the LLM-generated image prompt
- For GPT-Image models with reference: Uses `/v1/images/edits` endpoint
- For DALL-E or GPT-Image without reference: Uses `/v1/images/generations` endpoint
- Receives image URL or base64-encoded image
- Downloads/decodes and saves as PNG file locally

**Common Steps:**
- Images are saved in the same folder as other output files
- Reference images are analyzed to match design style exactly
- All images are accessible via the `/api/v1/images/{filename}` endpoint

## ⚠️ Limitations

**Social Media Content Generator** has the following limitations:

- **Image Generation Time**: Generating images for 10 slides takes ~1-2 minutes; single images take ~5-10 seconds
- **Cost**: API calls for image generation (check OpenRouter and OpenAI pricing)
- **Article Structure**: Works best with well-structured articles
- **Content Length**: Slides have character limits for optimal readability
- **Local Storage**: Images are saved locally; ensure sufficient disk space
- **List Detection**: List post detection works best with clearly numbered or bulleted lists
- **Reference Image Size**: Maximum 10MB file size for reference images
- **Reference Image Format**: Supports common image formats (PNG, JPG, JPEG)
- **Design Matching**: Reference image design matching depends on model's interpretation; results may vary
- **Reference Image Support**: 
  - GPT-Image models (gpt-image-1, gpt-image-1.5) support reference images via edits endpoint
  - DALL-E models (dall-e-3, dall-e-2) do not support reference images
  - OpenRouter models support reference images via vision capabilities

## 🤝 Contributing

Contributions to **Social Media Content Generator** are welcome! To add new features:

1. Create new tools or nodes in the agent
2. Update state definition if needed
3. Add to README documentation
4. Test thoroughly with the API endpoints

Please ensure your contributions align with the project's architecture and coding standards.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [OpenAI](https://openai.com) for content generation
- Image generation via [OpenRouter](https://openrouter.ai) using Google Gemini or [OpenAI](https://openai.com) DALL-E/GPT-Image
- API framework: [FastAPI](https://fastapi.tiangolo.com)
- Browser automation: [Playwright](https://playwright.dev)

## 🎨 Frontend Integration

The API is designed to be frontend-friendly with:

- **Consistent JSON Responses**: All responses include `success` flag and `request_id`
- **Image URLs**: Direct API URLs to access generated images (`image_url` field)
- **Error Handling**: Structured error responses with field-level validation details
- **Request Tracking**: Every request gets a unique ID for debugging
- **CORS Support**: Configured for cross-origin requests
- **Response Headers**: Includes `X-Request-ID` and `X-Process-Time` headers

### 📖 Complete Frontend Integration Guide

For comprehensive frontend integration documentation, see **[FRONTEND_INTEGRATION.md](./FRONTEND_INTEGRATION.md)**.

The guide includes:
- Complete API reference with all endpoints
- Code examples for React, Vue.js, and vanilla JavaScript
- TypeScript type definitions
- Error handling patterns
- Image handling and display
- Best practices and troubleshooting
- Common use cases and examples

### Quick Start Example

```javascript
// Generate carousel
const response = await fetch('http://localhost:8000/api/v1/generate-carousel-from-prompt', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your_api_key'
  },
  body: JSON.stringify({
    prompt: 'top 10 AI tools',
    max_slides: 10,
    image_provider: 'openai',
    image_model: 'dall-e-3'
  })
});

const data = await response.json();

if (data.success) {
  // Access images directly via image_url
  data.slides.forEach(slide => {
    console.log(`Slide ${slide.slide_number}: ${slide.image_url}`);
  });
} else {
  // Handle error
  console.error(data.error, data.errors);
}
```

### Image Access

```javascript
// Images are served via API endpoint
<img src={slide.image_url} alt={slide.title} />
// Example: http://localhost:8000/api/v1/images/slide_1_image.png
```

## 📞 Support

For issues or questions:
- Check the logs in `output/` directory
- Review API documentation at `/docs` endpoint
- Ensure API keys are valid
- Verify Playwright is installed correctly

## ⚠️ Important Notes

**Copyright**: Respect copyright when scraping articles. Ensure you have permission to use the content for carousel generation.

**API Keys**: Keep your API keys secure. Never commit them to version control.

---

---

**Social Media Content Generator** - Built with ❤️ using LangGraph and FastAPI

Start generating amazing social media content with AI today! 🎨

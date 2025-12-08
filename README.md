# Carousel Post Generator 🎨

A production-ready FastAPI application powered by LangGraph that automatically generates engaging carousel posts from article URLs. Uses AI to create slide content and generates custom images for each slide using Gemini via OpenRouter.

## 🌟 Features

- **Automatic Article Scraping**: Uses Playwright to scrape article content from any URL
- **AI-Powered Slide Generation**: Uses GPT-4o-mini to create engaging carousel slides from article content
- **Custom Image Generation**: Generates unique images for each slide using Gemini 2.5 Flash via OpenRouter
- **Local Image Storage**: Automatically saves generated images in a folder named after the post title
- **Structured Output**: Saves carousel posts in both Markdown and JSON formats with images in the same folder
- **Configurable Slides**: Customize the number of slides (3-15, default: 10)
- **LangGraph Workflow**: Production-ready agent following best practices
- **RESTful API**: FastAPI endpoints with security features

## 🔐 Security Features

- **API Key Authentication** - Secure header-based authentication
- **Rate Limiting** - Configurable per-minute rate limits
- **Input Validation** - Pydantic models for request validation
- **CORS Support** - Configurable cross-origin resource sharing
- **Error Handling** - Comprehensive error handling with proper HTTP status codes

## 🏗️ Architecture

Built using **LangGraph** and **FastAPI** with:
- **State Management**: TypedDict-based state tracking
- **Web Scraping**: Playwright for article content extraction
- **LLM Orchestration**: GPT-4o-mini for slide content generation
- **Image Generation**: Gemini 2.5 Flash via OpenRouter for custom images
- **Local Storage**: Automatic file organization in output folders

### Workflow

```
Article URL → Scrape Content → Generate Slides → Generate Images → Save Files → API Response
```

## 📋 Prerequisites

- Python 3.11+
- OpenAI API key (required for slide generation)
- OpenRouter API key (required for image generation)
- API key for authentication (optional, set `API_KEY` in `.env`)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
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
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional - Security
API_KEY=your_api_key_for_authentication
RATE_LIMIT_PER_MINUTE=10

# Optional - Server Configuration
PORT=8000
HOST=0.0.0.0
CORS_ORIGINS=*
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

### Health Check

```bash
curl http://localhost:8000/health
```

### Generate Carousel Post

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "max_slides": 10
  }'
```

### Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example Request

```json
{
  "url": "https://example.com/article",
  "max_slides": 8
}
```

### Example Response

```json
{
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
      "image_filename": "slide_1_hook_title.png",
      "image_relative_path": "./slide_1_hook_title.png"
    }
  ],
  "processing_status": "completed"
}
```

## 📊 Output

The API automatically saves results to the `output/` directory:

### File Structure

All files are saved in a folder named after the article title: `output/{article_title}/`

- **`carousel_{timestamp}.md`** - Human-readable markdown format with slide content and images
- **`carousel_{timestamp}.json`** - Machine-readable JSON format for further processing
- **`slide_1_*.png`** - Generated images for each slide
- **`slide_2_*.png`** - ...
- **`slide_N_*.png`** - Last slide image

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

- **Social Media Content**: Create engaging Instagram/LinkedIn carousels
- **Content Repurposing**: Convert blog posts into shareable carousel format
- **Educational Content**: Break down complex topics into digestible slides
- **Marketing**: Transform product articles into promotional carousels
- **Personal Branding**: Create thought leadership content from your articles

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
- Images are automatically saved as PNG files in the carousel folder
- Carousel-optimized prompts for consistent style
- Square orientation recommended for social media
- Images are saved with descriptive filenames based on their prompts

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

## 📝 Carousel Slide Structure

Each carousel post includes:

1. **Slide 1 (Hook)**: Attention-grabbing title slide that introduces the topic
2. **Slides 2-N-1 (Content)**: Key points and insights from the article
3. **Slide N (CTA)**: Call-to-action or conclusion slide

### Slide Components

Each slide contains:
- **Slide Number**: Sequential number (1-N)
- **Title**: Catchy title (max 60 characters)
- **Content**: Concise, valuable content (2-3 sentences, max 150 characters)
- **Image**: Custom-generated image relevant to the slide content
- **Image Prompt**: The prompt used to generate the image

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
├── main.py                              # FastAPI application
├── carousel_post_generator_agent.py     # LangGraph agent implementation
├── output/                              # Generated carousels and images
├── requirements.txt                     # Python dependencies
├── start_api.sh                         # Startup script
└── README.md                            # This file
```

## 🔄 How It Works

### 1. Article Scraping (Playwright)

The agent uses Playwright to:
- Navigate to the article URL
- Wait for dynamic content to load
- Extract article title, content, and metadata
- Parse headings and paragraphs
- Clean and structure the content

### 2. Slide Generation (GPT-4o-mini)

The LLM:
- Analyzes the article content
- Identifies key points and insights
- Creates engaging, concise slide content
- Generates detailed image prompts for each slide
- Ensures slides tell a cohesive story

### 3. Image Generation (Gemini via OpenRouter)

For each slide:
- Uses the LLM-generated image prompt
- Calls OpenRouter's Gemini 2.5 Flash image generation API
- Receives base64-encoded image
- Decodes and saves as PNG file locally
- Stores the local file path with the slide data
- Images are saved in the same folder as the markdown and JSON files

## ⚠️ Limitations

- **Image Generation Time**: Generating images for 10 slides takes ~1-2 minutes
- **Cost**: OpenRouter API calls for image generation (check OpenRouter pricing)
- **Article Structure**: Works best with well-structured articles
- **Content Length**: Slides have character limits for optimal readability
- **Local Storage**: Images are saved locally; ensure sufficient disk space

## 🤝 Contributing

Contributions are welcome! To add new features:

1. Create new tools or nodes in the agent
2. Update state definition if needed
3. Add to README documentation
4. Test thoroughly with the API endpoints

## 📄 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [OpenAI](https://openai.com) for content generation
- Image generation via [OpenRouter](https://openrouter.ai) and Gemini
- API framework: [FastAPI](https://fastapi.tiangolo.com)
- Browser automation: [Playwright](https://playwright.dev)

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

**Built with ❤️ using LangGraph and FastAPI**

Start generating amazing carousel posts today! 🎨
# Carousel-Post-Generator-AI-Agent

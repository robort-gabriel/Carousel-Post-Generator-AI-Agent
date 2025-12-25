# Frontend Integration Guide

Simple guide for frontend developers to integrate with the Social Media Content Generator FastAPI application.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
- [Request/Response Formats](#requestresponse-formats)
- [Error Handling](#error-handling)
- [Image Handling](#image-handling)

## Overview

The Social Media Content Generator API provides endpoints to generate:
- **Carousel Posts**: Multi-slide carousel posts from article URLs or text prompts
- **Single Informational Images**: One comprehensive image from article URLs
- **Infographics**: Generate infographics from text prompts or with reference images

All endpoints return direct image URLs that can be used in your frontend application.

## Getting Started

### Base URL

```
Development: http://localhost:8000
Production: https://api.yourdomain.com
```

### API Key

Get your API key from the backend administrator and include it in all requests via the `X-API-Key` header.

**⚠️ Security Note**: Never commit API keys to version control. Use environment variables or secure configuration management.

### Health Check

Before making requests, verify the API is available by calling `GET /health`. It should return `{"status": "healthy"}`.

## Authentication

All endpoints (except `/health` and `/api/v1/info`) require authentication via the `X-API-Key` header.

### Required Header

Include this header in all authenticated requests:

```
X-API-Key: your_api_key_here
```

### Content Type

For JSON requests, also include:

```
Content-Type: application/json
```

For file uploads (reference image endpoint), let the browser set the Content-Type automatically (it will include the boundary for multipart/form-data).

## API Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Check if the API is running and healthy.

**Authentication**: Not required

**Response**: 
```json
{
  "status": "healthy",
  "message": "API is running",
  "version": "1.0.0"
}
```

### 2. Get API Information

**Endpoint**: `GET /api/v1/info`

**Description**: Get API capabilities, supported models, and configuration.

**Authentication**: Not required

**Response**: Returns information about available endpoints, image providers, supported models, rate limits, and authentication requirements.

### 3. Generate Carousel from Article URL

**Endpoint**: `POST /api/v1/generate`

**Description**: Generate a carousel post from an article URL.

**Authentication**: Required

**Request Body**:
- `url` (required): Article URL to scrape
- `max_slides` (optional, default: 10): Number of slides to generate (3-15)
- `username` (optional): Social media username to display
- `tagline` (optional): Brand tagline to display
- `title` (optional): Custom title to override article title
- `font_name` (optional): Font name for slides
- `background_info` (optional): Background description
- `color_schema` (optional): Color scheme description
- `extra_instructions` (optional): Additional instructions for customization
- `image_provider` (optional): "openrouter" or "openai"
- `image_model` (optional): Model name (e.g., "dall-e-3", "google/gemini-2.5-flash-image")

**Response**: Returns carousel with slides array. Each slide contains `image_url` field for direct image access.

### 4. Generate Carousel from Prompt

**Endpoint**: `POST /api/v1/generate-carousel-from-prompt`

**Description**: Generate a carousel post directly from a text prompt (no URL needed).

**Authentication**: Required

**Request Body**:
- `prompt` (required): Text prompt describing what to create (e.g., "top 10 free AI tools")
- `max_slides` (optional, default: 10): Number of slides to generate (3-15)
- `username` (optional): Social media username to display
- `tagline` (optional): Brand tagline to display
- `font_name` (optional): Font name for slides
- `background_info` (optional): Background description
- `color_schema` (optional): Color scheme description
- `extra_instructions` (optional): Additional instructions for customization
- `image_provider` (optional): "openrouter" or "openai"
- `image_model` (optional): Model name

**Response**: Returns carousel with slides array. Each slide contains `image_url` field for direct image access.

### 5. Generate Single Informational Image

**Endpoint**: `POST /api/v1/generate-single-image`

**Description**: Generate a single informational image from an article URL.

**Authentication**: Required

**Request Body**:
- `url` (required): Article URL to scrape
- `username` (optional): Social media username to display
- `tagline` (optional): Brand tagline to display
- `title` (optional): Custom title to override article title
- `font_name` (optional): Font name
- `background_info` (optional): Background description
- `color_schema` (optional): Color scheme description
- `extra_instructions` (optional): Additional instructions
- `image_provider` (optional): "openrouter" or "openai"
- `image_model` (optional): Model name

**Response**: Returns single image with `image_url` field for direct image access.

### 6. Generate Infographic from Prompt

**Endpoint**: `POST /api/v1/generate-infographic-from-prompt`

**Description**: Generate an infographic from a text prompt.

**Authentication**: Required

**Request Body**:
- `prompt` (required): Text prompt (e.g., "top 10 free AI tools", "how to use React hooks")
- `username` (optional): Social media username to display
- `tagline` (optional): Brand tagline to display
- `font_name` (optional): Font name
- `background_info` (optional): Background description
- `color_schema` (optional): Color scheme description
- `extra_instructions` (optional): Additional instructions
- `image_provider` (optional): "openrouter" or "openai"
- `image_model` (optional): Model name

**Response**: Returns infographic with `image_url` field for direct image access.

### 7. Generate Infographic with Reference Image

**Endpoint**: `POST /api/v1/generate-infographic-with-reference`

**Description**: Generate an infographic matching a reference image design.

**Authentication**: Required

**Request Format**: `multipart/form-data` (file upload)

**Form Fields**:
- `reference_image` (required): Image file (PNG, JPG, JPEG, max 10MB)
- `prompt` (required): Text prompt describing content
- `username` (optional): Social media username
- `tagline` (optional): Brand tagline
- `font_name` (optional): Font name
- `background_info` (optional): Background description
- `color_schema` (optional): Color scheme description
- `extra_instructions` (optional): Additional instructions
- `image_provider` (optional): "openrouter" or "openai"
- `image_model` (optional): Model name (use "gpt-image-1" or "gpt-image-1.5" for reference support)

**Response**: Returns infographic with `image_url` field for direct image access.

**Note**: For reference images, use GPT-Image models (gpt-image-1, gpt-image-1.5) or OpenRouter models. DALL-E models do not support reference images.

### 8. Serve Generated Images

**Endpoint**: `GET /api/v1/images/{filename}`

**Description**: Access generated images directly via URL.

**Authentication**: Not required

**Usage**: Use the `image_url` field from any response. Images are served directly and can be used in `<img>` tags or downloaded.

## Request/Response Formats

### Success Response Structure

All successful responses include:
- `success`: `true`
- `status`: Response status string
- `request_id`: Unique request identifier for tracking

### Carousel Response

```json
{
  "success": true,
  "status": "success",
  "url": "article_url_or_empty",
  "article_title": "Title of the content",
  "total_slides": 10,
  "slides": [
    {
      "slide_number": 1,
      "title": "Slide title",
      "content": "Slide content",
      "image_url": "http://localhost:8000/api/v1/images/slide_1_image.png",
      "image_filename": "slide_1_image.png"
    }
  ],
  "processing_status": "completed",
  "request_id": "unique-request-id"
}
```

### Single Image Response

```json
{
  "success": true,
  "status": "success",
  "url": "article_url",
  "article_title": "Article title",
  "title": "Image title",
  "image_url": "http://localhost:8000/api/v1/images/image.png",
  "request_id": "unique-request-id"
}
```

### Infographic Response

```json
{
  "success": true,
  "status": "success",
  "prompt": "user prompt",
  "type": "list",
  "title": "Infographic title",
  "content": "Infographic content",
  "image_url": "http://localhost:8000/api/v1/images/infographic.png",
  "image_filename": "infographic.png",
  "request_id": "unique-request-id"
}
```

### Error Response

```json
{
  "success": false,
  "error": "Error type",
  "detail": "Detailed error message",
  "error_type": "validation_error",
  "errors": [
    {
      "field": "url",
      "message": "field required",
      "type": "value_error.missing"
    }
  ],
  "request_id": "unique-request-id"
}
```

## Error Handling

### HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request data (validation errors)
- `401 Unauthorized`: Missing or invalid API key
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error during processing

### Error Response Structure

All errors return JSON with:
- `success`: `false`
- `error`: Error type
- `detail`: Human-readable error message
- `error_type`: Type of error (e.g., "validation_error")
- `errors`: Array of field-level validation errors (if applicable)
- `request_id`: Request ID for tracking

### Rate Limiting

The API implements rate limiting per IP address. When rate limited:
- Status code: `429`
- Check `Retry-After` header for when to retry
- Implement exponential backoff for retries

### Common Errors

1. **Missing API Key**: Include `X-API-Key` header
2. **Invalid API Key**: Verify API key is correct
3. **Validation Error**: Check request body matches required format
4. **Rate Limit**: Wait before retrying or reduce request frequency
5. **Network Error**: Check API server is running and accessible

## Image Handling

### Using Image URLs

All responses include `image_url` fields that point directly to generated images. Use these URLs directly:

- In HTML: `<img src="{image_url}" alt="description" />`
- In React: `<img src={slide.image_url} alt={slide.title} />`
- For downloads: Use the URL in a download link or fetch and save

### Image URL Format

Image URLs follow this pattern:
```
http://localhost:8000/api/v1/images/{filename}
```

### Image Loading

- Images are generated asynchronously and may take 5-10 seconds per image
- For carousels with multiple slides, generation can take 1-2 minutes
- Show loading states while waiting for generation
- Handle image load errors gracefully

### Image Formats

- All images are served as PNG files
- Images are optimized for social media (typically 1080x1080 or 1080x1350)

## Implementation Checklist

1. ✅ Set up base URL and API key configuration
2. ✅ Implement authentication header (`X-API-Key`)
3. ✅ Add error handling for all HTTP status codes
4. ✅ Handle rate limiting (429 errors)
5. ✅ Display loading states for long-running operations
6. ✅ Use `image_url` fields from responses to display images
7. ✅ Handle validation errors and display field-level messages
8. ✅ Store `request_id` for debugging and support requests
9. ✅ Implement retry logic with exponential backoff
10. ✅ Test all endpoints with your frontend framework

## Additional Resources

- **Interactive API Documentation**: Visit `/docs` endpoint for Swagger UI
- **Alternative Documentation**: Visit `/redoc` endpoint for ReDoc
- **API Capabilities**: Call `/api/v1/info` to get current API configuration

## Support

For issues or questions:
- Check error responses for detailed messages
- Use `request_id` from responses to track requests in backend logs
- Verify API keys and configuration
- Contact backend team with `request_id` for debugging

---

**Ready to integrate! 🚀**

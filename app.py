import time
from fastapi import FastAPI, Query, HTTPException, Security, status, Depends
from fastapi.security import APIKeyHeader
from typing import Optional
import uvicorn

from config import setup_logging, load_environment, configure_gemini, get_auth_key
from models import SuccessResponse, ErrorResponse, HealthResponse, RecipeResponse
from utils import get_platform
from scraper import (
    try_video_extraction,
    try_recipe_scraper,
    try_json_ld,
    try_gemini_extraction,
    translate_recipe,
)

load_environment()
logger = setup_logging()
gemini_api_key = configure_gemini()
AUTH_KEY = get_auth_key()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not AUTH_KEY:
        # If no AUTH_KEY is set in env, allow access (or you could decide to block)
         logger.warning("No AUTH_KEY configured, API is unsecured")
         return None
    
    if api_key_header == AUTH_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )

app = FastAPI(
    title="Unified Recipe Scraper API",
    version="2.2.0",
    description="A recipe scraping API that extracts and translates recipe data from websites, TikTok, and YouTube.",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 60)
    logger.info("Recipe Scraper API Starting...")
    logger.info(f"Version: 2.2.0")
    logger.info(f"Gemini API: {'Configured' if gemini_api_key else 'NOT CONFIGURED'}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information"""
    logger.info("=" * 60)
    logger.info("Recipe Scraper API Shutting Down...")
    logger.info("=" * 60)


@app.get(
    "/",
    summary="Health Check",
    description="Basic health check endpoint to verify API is running",
)
async def root():
    """Root endpoint - Health check"""
    return {
        "status": "healthy",
        "app": "Unified Recipe Scraper API",
        "version": "2.2.0",
        "platforms": ["website", "tiktok", "youtube"],
        "description": "API for extracting and translating recipes from websites, TikTok, and YouTube",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health",
    description="Detailed system health check with configuration status",
)
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "gemini_configured": bool(gemini_api_key),
        "endpoints": {
            "scrape": "/scrape?url=<recipe-url>&language=<target-language>",
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.get(
    "/scrape",
    response_model=SuccessResponse,
    dependencies=[Depends(get_api_key)],
    responses={
        200: {"description": "Recipe successfully scraped and formatted"},
        500: {
            "description": "Failed to extract recipe",
            "model": ErrorResponse,
        },
    },
    summary="Scrape and Translate Recipe",
    description="""Extracts and translates recipe data from any website, TikTok, or YouTube video.

**Extraction Methods:**
- **Videos (TikTok/YouTube):** Gemini AI analyzes video content.
- **Websites:** Uses a 3-step fallback (recipe-scrapers library, JSON-LD extraction, Gemini AI).

**Translation:**
- Provide a `language` query parameter (e.g., `arabic`, `spanish`, `french`) to translate the recipe.
- If omitted, defaults to `english`.
- Translation only occurs if the detected language differs from the target language.
""",
)
async def scrape_recipe(
    url: str = Query(
        ...,
        description="URL of the recipe to scrape (website, TikTok video, or YouTube video)",
        example="https://www.allrecipes.com/recipe/22180/waffles-i/",
    ),
    language: Optional[str] = Query(
        "english",
        description="Target language for the recipe output (e.g., 'english', 'arabic', 'spanish')",
        example="arabic",
    ),
):
    """Scrape and translate a recipe from any website, TikTok, or YouTube."""
    start_time = time.time()
    logger.info(f"Scrape request for URL: {url} | Target language: {language}")

    platform = get_platform(url)
    logger.info(f"Platform: {platform}")

    try:
        if platform in ["tiktok", "youtube"]:
            logger.info(f"Processing {platform} video...")
            data = try_video_extraction(url, platform, gemini_api_key)
            source = f"{platform}-video"
        else:
            # Website scraping with 3-step fallback
            logger.info("Step 1/3: Trying recipe-scrapers...")
            try:
                data = try_recipe_scraper(url)
                source = "recipe-scraper"
            except Exception:
                logger.info("Step 2/3: Trying JSON-LD extraction...")
                try:
                    data = try_json_ld(url, gemini_api_key)
                    source = "json-ld"
                except Exception:
                    logger.info("Step 3/3: Trying Gemini extraction...")
                    data = try_gemini_extraction(url, gemini_api_key)
                    source = "gemini"

        elapsed_scrape = time.time() - start_time
        logger.info(f"SUCCESS via {source} ({elapsed_scrape:.2f}s)")

        # Translate if target language differs from recipe language
        if language.lower() != "english":
            logger.info(f"Attempting translation to {language}...")
            data = translate_recipe(data, language, gemini_api_key)
            source = f"{source}-translated-{language}"

        total_elapsed = time.time() - start_time
        return SuccessResponse(
            success=True,
            source=source,
            processing_time=round(total_elapsed, 3),
            data=RecipeResponse(**data),
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"ALL METHODS FAILED for {url} ({elapsed:.2f}s): {type(e).__name__}"
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                success=False,
                message="Cannot scrape recipe from this site",
                error_type=type(e).__name__,
            ).dict(),
        )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=6000,
        reload=True,
        log_level="info",
    )

import logging
import os
from dotenv import load_dotenv
import google.generativeai as genai


def setup_logging():
    """Configures the logging for the application."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
    )
    # Silence watchfiles logs
    logging.getLogger("watchfiles").setLevel(logging.CRITICAL)
    logging.getLogger("watchfiles.main").setLevel(logging.CRITICAL)
    logging.getLogger("watchfiles.watcher").setLevel(logging.CRITICAL)
    return logging.getLogger(__name__)


def load_environment():
    """Loads environment variables from .env file."""
    load_dotenv()
    logger = logging.getLogger(__name__)
    logger.info("Loading environment variables from .env file...")


def configure_gemini():
    """Configures the Gemini API."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    logger = logging.getLogger(__name__)
    logger.debug("Checking for GEMINI_API_KEY environment variable...")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        logger.info(
            f"Gemini API configured successfully (key length: {len(gemini_api_key)})"
        )
        return gemini_api_key
    else:
        logger.warning("GEMINI_API_KEY not found in environment variables")
        return None


# Constants
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

UNIFIED_RECIPE_FORMAT = {
    "title": "",
    "description": "",
    "prep_time": 0,
    "cook_time": 0,
    "total_time": 0,
    "yields": 0,
    "ingredients": [],
    "instructions": [],
    "image": {"url": "", "key": None},
    "url": "",
    "host": "",
}

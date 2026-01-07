import logging
import json
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from recipe_scrapers import scrape_me
from typing import Dict, Optional
from urllib.parse import urlparse

from config import HEADERS, UNIFIED_RECIPE_FORMAT
from utils import parse_time_to_minutes, parse_servings_to_int
from s3_upload import upload_image_if_configured

logger = logging.getLogger(__name__)


def detect_language(text: str, gemini_api_key: str) -> str:
    """Detect the language of the given text using Gemini AI."""
    if not gemini_api_key:
        return "unknown"

    prompt = f"""Detect the language of the following text and return ONLY the language name in English (e.g., "english", "arabic", "spanish", "french").
    
Text: {text[:500]}

Language:"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        detected_language = response.text.strip().lower()
        return detected_language
    except Exception as e:
        logger.warning(f"Language detection failed: {str(e)}")
        return "unknown"


def get_video_metadata(url: str, platform: str) -> Dict[str, str]:
    """Fetch video metadata using oEmbed API."""
    metadata = {"title": "", "author": "", "thumbnail_url": ""}

    try:
        if platform == "tiktok":
            oembed_url = (
                f"https://www.tiktok.com/oembed?url={requests.utils.quote(url)}"
            )
        elif platform == "youtube":
            oembed_url = (
                f"https://www.youtube.com/oembed?url={requests.utils.quote(url)}"
            )
        else:
            return metadata

        response = requests.get(oembed_url, timeout=10)
        if response.ok:
            oembed_data = response.json()
            metadata["title"] = oembed_data.get("title", "")
            metadata["author"] = oembed_data.get("author_name", "")
            metadata["thumbnail_url"] = oembed_data.get("thumbnail_url", "")
    except Exception as e:
        logger.warning(f"Failed to fetch {platform} metadata: {str(e)}")

    return metadata


def try_video_extraction(url: str, platform: str, gemini_api_key: str) -> Dict:
    """Extract recipe from TikTok or YouTube video using Gemini AI."""
    if not gemini_api_key:
        raise ValueError("Gemini API key is required for video extraction")

    metadata = get_video_metadata(url, platform)
    thumbnail_url = metadata.get("thumbnail_url", "")
    uploaded_image = upload_image_if_configured(thumbnail_url)

    platform_name = platform.capitalize()
    title_author_info = ""
    if metadata["title"] and metadata["author"]:
        title_author_info = (
            f'titled "{metadata["title"]}" by author "{metadata["author"]}"'
        )

    prompt = f"""Extract a recipe from the {platform_name} video at: {url}
The video is {title_author_info}.

Return ONLY valid JSON in this EXACT format:
{json.dumps(UNIFIED_RECIPE_FORMAT, indent=2)}

Instructions:
- title: Recipe name
- description: Short summary
- ingredients: List with quantities (e.g., ["1 onion", "2 cups flour"])
- instructions: Step-by-step list
- prep_time, cook_time, total_time: INTEGER minutes (use 0 if not mentioned)
- yields: INTEGER servings (use 0 if not mentioned)
- image: {{"url": "{uploaded_image['url']}", "key": "{uploaded_image['key']}"}}
- url: "{url}"
- host: "{platform_name}"

Return ONLY the JSON object."""

    try:
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={"response_mime_type": "application/json"},
        )
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        validated_result = validate_recipe_structure(result)

        validated_result["url"] = url
        validated_result["host"] = platform_name
        validated_result["image"] = {
            "url": uploaded_image["url"],
            "key": uploaded_image["key"],
        }

        return validated_result
    except Exception as e:
        logger.error(f"Video extraction failed: {str(e)}")
        raise


def format_recipe_scrapers_data(scraper_data: Dict) -> Dict:
    """Format recipe-scrapers data to unified format."""
    unified = UNIFIED_RECIPE_FORMAT.copy()

    unified["title"] = scraper_data.get("title", "")
    unified["description"] = scraper_data.get("description", "")
    unified["prep_time"] = parse_time_to_minutes(scraper_data.get("prep_time", ""))
    unified["cook_time"] = parse_time_to_minutes(scraper_data.get("cook_time", ""))
    unified["total_time"] = parse_time_to_minutes(scraper_data.get("total_time", ""))

    yields_str = scraper_data.get("yields", "") or scraper_data.get("servings", "")
    unified["yields"] = parse_servings_to_int(yields_str)

    image_url = scraper_data.get("image", "")
    uploaded_image = upload_image_if_configured(image_url)
    unified["image"] = {"url": uploaded_image["url"], "key": uploaded_image["key"]}

    unified["url"] = scraper_data.get("url", "")
    unified["host"] = scraper_data.get("host", "")

    ingredients = scraper_data.get("ingredients", [])
    if isinstance(ingredients, list):
        unified["ingredients"] = [
            str(ingredient) for ingredient in ingredients if ingredient
        ]
    else:
        unified["ingredients"] = [str(ingredients)] if ingredients else []

    instructions = scraper_data.get("instructions", "")
    if isinstance(instructions, list):
        unified["instructions"] = [
            str(instruction) for instruction in instructions if instruction
        ]
    elif isinstance(instructions, str):
        unified["instructions"] = [
            inst.strip() for inst in instructions.split("\n") if inst.strip()
        ]
    else:
        unified["instructions"] = []

    return unified


def format_with_gemini(
    raw_data: Dict, source: str, url: str, gemini_api_key: str
) -> Dict:
    """Use Gemini to format recipe data into unified format."""
    if not gemini_api_key:
        raise ValueError("Gemini API is required for formatting")

    prompt = f"""Convert this recipe data to EXACT JSON format.
Return ONLY valid JSON with proper syntax.

REQUIRED FORMAT:
{json.dumps(UNIFIED_RECIPE_FORMAT, indent=2)}

SOURCE: {source}
URL: {url}

RAW DATA:
{json.dumps(raw_data, indent=2)}

CRITICAL RULES:
- Return ONLY the JSON object, no explanations
- Ensure all strings are properly quoted
- Times: INTEGER minutes (e.g., "1 hour 30 min" -> 90)
- Yields: INTEGER servings (e.g., "4-6 servings" -> 4)

INGREDIENTS RULES (VERY IMPORTANT):
- ONLY include actual ingredients with quantities
- DO NOT include section headers like "FOR THE SAUCE", "FOR THE TOPPING", "FOR THE SALAD", etc.
- DO NOT include blank lines or separators
- Each ingredient must have a quantity (e.g., "1 cup flour", "2 tablespoons sugar", "1/2 teaspoon salt")
- Format: ["quantity + unit + ingredient", "quantity + unit + ingredient", ...]
- Example: ["1 cup quinoa", "2 chicken breasts", "1/2 green bell pepper"]

INSTRUCTIONS RULES (VERY IMPORTANT):
- ONLY include actual cooking steps
- DO NOT include section headers like "FOR THE SAUCE", "FOR THE TOPPING", "FOR THE SALAD", etc.
- DO NOT include blank lines or separators
- Remove notes like "NOTE:", "TIP:", "CHEF'S NOTE:", etc.
- Each step should be a clear cooking action
- Combine steps from different sections into one continuous list
- Example: ["Add ingredients to pan and heat", "Bring to simmer for 2 minutes", "Assemble bowls with quinoa and toppings"]

OTHER RULES:
- image: {{"url": "...", "key": null}} or extract from raw data
- Use 0 for missing integers, "" for strings, [] for lists

Return ONLY valid JSON:"""

    try:
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        )
        response = model.generate_content(prompt)

        # Clean the response text
        response_text = response.text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = (
                response_text.split("\n", 1)[1]
                if "\n" in response_text
                else response_text[3:]
            )
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

        # Try to parse JSON
        try:
            formatted_recipe = json.loads(response_text)
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON parsing failed in format_with_gemini: {str(json_err)}")
            logger.error(f"Response text (first 500 chars): {response_text[:500]}")
            raise ValueError(f"Gemini returned invalid JSON: {str(json_err)}")

        return validate_recipe_structure(formatted_recipe)
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Gemini formatting failed: {str(e)}")
        raise


def validate_recipe_structure(recipe_data: Dict) -> Dict:
    """Ensure recipe data matches UNIFIED_RECIPE_FORMAT with proper types."""
    validated = UNIFIED_RECIPE_FORMAT.copy()

    for key in validated:
        if key not in recipe_data:
            continue

        if isinstance(validated[key], list):
            if isinstance(recipe_data[key], list):
                validated[key] = recipe_data[key]
            else:
                validated[key] = [str(recipe_data[key])] if recipe_data[key] else []

        elif isinstance(validated[key], int):
            if isinstance(recipe_data[key], int):
                validated[key] = recipe_data[key]
            elif isinstance(recipe_data[key], str):
                if key in ["prep_time", "cook_time", "total_time"]:
                    validated[key] = parse_time_to_minutes(recipe_data[key])
                elif key == "yields":
                    validated[key] = parse_servings_to_int(recipe_data[key])
                else:
                    try:
                        validated[key] = int(recipe_data[key])
                    except (ValueError, TypeError):
                        validated[key] = 0
            else:
                validated[key] = 0

        elif isinstance(validated[key], dict):
            if isinstance(recipe_data[key], dict):
                validated[key] = recipe_data[key]
            elif isinstance(recipe_data[key], str):
                # Handle old format where image was a string
                validated[key] = {"url": recipe_data[key], "key": None}

        else:
            validated[key] = (
                str(recipe_data[key]) if recipe_data[key] is not None else ""
            )

    return validated


def try_recipe_scraper(url: str) -> Dict:
    """Attempt to scrape using recipe-scrapers package."""
    try:
        scraper = scrape_me(url)
        result = scraper.to_json()
        logger.debug(f"recipe-scrapers raw data: {result}")
        return format_recipe_scrapers_data(result)
    except Exception as e:
        logger.warning(f"recipe-scrapers failed: {e}", exc_info=True)
        raise


def is_recipe_data(data: dict) -> bool:
    """Check if a dictionary contains recipe-like data."""
    item_type = data.get("@type", "")
    if item_type in ["Recipe", "FoodRecipe"] or "recipe" in str(item_type).lower():
        return True

    recipe_fields = ["recipeIngredient", "ingredients", "recipeInstructions"]
    return any(field in data for field in recipe_fields)


def try_json_ld(url: str, gemini_api_key: str) -> Dict:
    """Extract recipe via JSON-LD schema from HTML."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.info(f"HTTP request failed: {type(e).__name__}")
        raise

    soup = BeautifulSoup(response.text, "html.parser")
    scripts = soup.find_all("script", type="application/ld+json")

    recipe_data = None
    for script in scripts:
        try:
            data = json.loads(script.string)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and is_recipe_data(item):
                        recipe_data = item
                        break

            elif isinstance(data, dict):
                item_type = data.get("@type", "")

                if (
                    item_type in ["Recipe", "FoodRecipe"]
                    or "recipe" in item_type.lower()
                ):
                    recipe_data = data

                elif "@graph" in data and isinstance(data["@graph"], list):
                    for item in data["@graph"]:
                        if isinstance(item, dict) and is_recipe_data(item):
                            recipe_data = item
                            break

                elif is_recipe_data(data):
                    recipe_data = data

            if recipe_data:
                # Single debug log to show extracted structure
                logger.debug(f"JSON-LD structure: {json.dumps(recipe_data, indent=2)}")
                break

        except (json.JSONDecodeError, Exception):
            continue

    if not recipe_data:
        raise ValueError("No valid JSON-LD recipe found")

    formatted_recipe = format_with_gemini(recipe_data, "json-ld", url, gemini_api_key)

    if formatted_recipe.get("image"):
        image_url = (
            formatted_recipe["image"].get("url")
            if isinstance(formatted_recipe["image"], dict)
            else formatted_recipe["image"]
        )
        uploaded_image = upload_image_if_configured(image_url)
        formatted_recipe["image"] = {
            "url": uploaded_image["url"],
            "key": uploaded_image["key"],
        }

    return formatted_recipe


def try_gemini_extraction(url: str, gemini_api_key: str) -> Dict:
    """Fallback: Use Gemini to extract recipe from page content."""
    if not gemini_api_key:
        raise ValueError("Gemini API key is required for extraction")

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.info(f"HTTP request failed: {type(e).__name__}")
        raise

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text_content = soup.get_text(separator=" ", strip=True)
    text_content = " ".join(text_content.split())  # Clean up whitespace
    truncated_content = text_content[:12000]

    prompt = f"""Extract recipe information from this webpage content and return VALID JSON format.

CRITICAL: You MUST return ONLY valid JSON with proper syntax. No markdown, no explanations, no extra text.

REQUIRED FORMAT:
{json.dumps(UNIFIED_RECIPE_FORMAT, indent=2)}

WEBPAGE CONTENT:
{truncated_content}

URL: {url}

RULES:
- Return ONLY the JSON object, nothing else
- Ensure all strings are properly quoted
- Ensure all arrays have proper commas between elements
- Times: INTEGER minutes (e.g., "1 hour 30 min" -> 90)
- Yields: INTEGER servings (e.g., "4-6 servings" -> 4)

INGREDIENTS RULES (VERY IMPORTANT):
- ONLY include actual ingredients with quantities
- DO NOT include section headers like "FOR THE SAUCE", "FOR THE TOPPING", "FOR THE SALAD", etc.
- DO NOT include blank lines or separators
- Each ingredient must have a quantity (e.g., "1 cup flour", "2 tablespoons sugar", "1/2 teaspoon salt")
- Format: ["quantity + unit + ingredient", "quantity + unit + ingredient", ...]
- Example: ["1 cup quinoa", "2 chicken breasts", "1/2 green bell pepper"]

INSTRUCTIONS RULES (VERY IMPORTANT):
- ONLY include actual cooking steps
- DO NOT include section headers like "FOR THE SAUCE", "FOR THE TOPPING", "FOR THE SALAD", etc.
- DO NOT include blank lines or separators
- Remove notes like "NOTE:", "TIP:", "CHEF'S NOTE:", etc.
- Each step should be a clear cooking action
- Combine steps from different sections into one continuous list
- Example: ["Add ingredients to pan and heat", "Bring to simmer for 2 minutes", "Assemble bowls with quinoa and toppings"]

OTHER RULES:
- image: {{"url": "...", "key": null}}
- Use 0 for missing integers, "" for strings, [] for lists

Return ONLY valid JSON:"""

    try:
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,  # Lower temperature for more consistent output
            },
        )
        response = model.generate_content(prompt)

        # Clean the response text
        response_text = response.text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            # Remove ```json or ``` at start
            response_text = (
                response_text.split("\n", 1)[1]
                if "\n" in response_text
                else response_text[3:]
            )
            # Remove ``` at end
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

        # Try to parse JSON
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON parsing failed: {str(json_err)}")
            logger.error(f"Response text (first 500 chars): {response_text[:500]}")
            raise ValueError(f"Gemini returned invalid JSON: {str(json_err)}")

        validated_result = validate_recipe_structure(result)

        # Upload image if found
        if validated_result.get("image"):
            image_url = (
                validated_result["image"].get("url")
                if isinstance(validated_result["image"], dict)
                else validated_result["image"]
            )
            if image_url:
                uploaded_image = upload_image_if_configured(image_url)
                validated_result["image"] = {
                    "url": uploaded_image["url"],
                    "key": uploaded_image["key"],
                }

        return validated_result
    except ValueError:
        # Re-raise ValueError with our custom message
        raise
    except Exception as e:
        logger.error(f"Gemini extraction failed: {str(e)}")
        raise


def translate_recipe(
    recipe_data: Dict, target_language: str, gemini_api_key: str
) -> Dict:
    """Translate recipe to target language if different from current language."""
    if not gemini_api_key:
        raise ValueError("Gemini API key is required for translation")

    # Detect current language from recipe title and description
    sample_text = f"{recipe_data.get('title', '')} {recipe_data.get('description', '')}"
    current_language = detect_language(sample_text, gemini_api_key)

    # Skip translation if languages match
    if current_language.lower() == target_language.lower():
        logger.info(f"Recipe is already in {target_language}, skipping translation")
        return recipe_data

    translatable_data = {
        "title": recipe_data.get("title", ""),
        "description": recipe_data.get("description", ""),
        "ingredients": recipe_data.get("ingredients", []),
        "instructions": recipe_data.get("instructions", []),
    }

    prompt = f"""Translate this JSON to {target_language}.

Rules:
- Keep JSON structure identical
- Do NOT translate field names (keys)
- Translate values only
- Return ONLY valid JSON

Original JSON:
{json.dumps(translatable_data, indent=2)}

Translated JSON:"""

    try:
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={"response_mime_type": "application/json"},
        )
        response = model.generate_content(prompt)
        translated_data = json.loads(response.text)

        translated_recipe = recipe_data.copy()
        translated_recipe.update(translated_data)

        logger.info(
            f"Successfully translated recipe from {current_language} to {target_language}"
        )
        return translated_recipe
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        raise

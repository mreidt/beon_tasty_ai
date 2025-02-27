import logging
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self, api_key):
        self.api_key = api_key

    def __get_prompt(self, title, ingredients):
        prompt = f"A beautiful photo of {title} with ingredients: {', '.join(ingredients)}."
        return prompt

    def generate_image(self, meal: dict, model = "dall-e-2"):
        """Generate an image using OpenAI's DALL-E API."""
        openai.api_key = self.api_key
        try:
            response = openai.images.generate(
                prompt=self.__get_prompt(meal.get("title"), meal.get("ingredients")),
                model=model,
                n=1,
                size="512x512"
            )
            image_url = response.data[0].url
            return image_url
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None

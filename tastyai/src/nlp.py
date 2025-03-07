import json
import logging

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_openai import ChatOpenAI
from translator import Translator
from user_profile import UserProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLP:
    def __init__(self, model_name="gpt-4o-mini", openai_api_key=None):
        self.openai_key = openai_api_key
        self.llm = ChatOpenAI(model_name=model_name, temperature=0, api_key=openai_api_key)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an AI assistant that extracts meal preferences from user queries. Extract dietary "
                        "preferences, sugar content, sharing suitability, any specific ingredients mentioned and "
                        "excluded ingredients. Pay close attention to nuances in sugar content descriptions. For sugar "
                        "content, use 'sugar_free' if the user does not wants sugar, 'low' if the user wants less "
                        "sugar, 'high' if they want more sugar, and 'normal' only if sugar content is not specifically "
                        "mentioned or emphasized. You should identify if the user is asking for a recipe based on the "
                        "prompt. The user can ask things like: i want a meal, i want a dessert, i want a dinner, and "
                        "things like that. "
                        "If the user is not asking for a recipe, set 'is_recipe_request' to false."
                        "Return the information in "
                        "a JSON format with the following structure (return all translated to english, no matter the "
                        "language that the user used): "
                        '{{"dietary": [], "sugar_content": "low, normal, high", "sharing": bool, '
                        '"ingredients": [], "excluded_ingredients": [], '
                        '"is_recipe_request": bool}}'
                    ),
                ),
                ("human", "{query}"),
            ]
        )
        self.chain = RunnableSequence(self.prompt | self.llm)

    def process_user_input(self, input_text):
        translator = Translator(openai_key=self.openai_key)
        language = translator.detect_language(input_text)
        if language != "english":
            input_text = translator.translate(input_text)
        result = self.chain.invoke({"query": input_text})
        try:
            content = result.content
            preferences = json.loads(content)
            logger.info(f"Preferences: {preferences}")

            user_profile = UserProfile(preferences)
            user_profile.__setattr__("language", language)
            logger.info(f"User profile: {user_profile}")
            return user_profile
        except json.JSONDecodeError:
            logger.error("Error: Unable to parse JSON response from ChatGPT")
            return None

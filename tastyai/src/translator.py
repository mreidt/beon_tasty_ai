import json
import logging

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_openai import ChatOpenAI
from langdetect import detect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Translator:
    def __init__(self, openai_key, language="english"):
        self.openai_key = openai_key
        self.language = language
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=openai_key)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an AI assistant that translates text from one language to another. You should be able "
                        "to translate text from english to spanish (and vice versa) and english to portuguese (and "
                        f"vice versa). Return the translated text in the following language: '''{self.language}'''. "
                        "Return the information as a simple text string."
                    ),
                ),
                ("human", "{query}"),
            ]
        )

    def translate(self, text):
        chain = RunnableSequence(self.prompt | self.llm)
        result = chain.invoke({"query": text})
        try:
            content = result.content
            logger.info(f"Translation: {content}")
            return content
        except json.JSONDecodeError:
            logger.error("Error: Unable to parse JSON response from ChatGPT")
            return None

    def detect_language(self, text):
        language = detect(text)
        logger.info(f"Detected language: {language} on text {text}")
        if language == "es":
            return "spanish"
        elif language == "pt":
            return "portuguese"
        else:
            return "english"

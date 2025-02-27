from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import logging
import json
from langchain.schema.runnable import RunnableSequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Translator:
    def __init__(self, openai_key, language="english"):
        self.openai_key = openai_key
        self.language = language
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=openai_key)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                    "You are an AI assistant that translates text from one language to another. You should be able to "
                    "translate text from english to spanish and english to portuguese. Return the translated text in "
                    f"the following language: '''{self.language}'''. Return the information as a simple text string."
                )
            ),
            ("human", "{query}")
        ])
    
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

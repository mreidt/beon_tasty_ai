import streamlit as st
from tastyai.vectorizer import Vectorizer
import logging
from tastyai.nlp import NLP
from tastyai.recommendation import Recommendation
from tastyai.image_generator import ImageGenerator
from PIL import Image
import requests
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting TastyAI chat app")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

st.title("TastyAI")
with st.form("my_form"):
    text = st.text_area(
        "Enter text (we can answer in portuguese, spanish and english):",
        value="I want a sugared meal that does not contain too much sugar and that I can share with my husband.",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):

        vectorizer = Vectorizer('./tastyai/dataset/full_dataset.csv')
        nlp = NLP()
        image_generator = ImageGenerator(openai_api_key)
        recommendation = Recommendation(vectorizer, openai_api_key)

        user_profile = nlp.process_user_input(text)
        if not user_profile.is_recipe_request:
            if user_profile.language == "spanish":
                st.warning("¡Por favor, pide una receta!", icon="⚠")
            elif user_profile.language == "portuguese":
                st.warning("Por favor, peça uma receita!", icon="⚠")
            else:
                st.warning("Please ask for a recipe!", icon="⚠")
            st.stop()
        recommendations = recommendation.get_meal_recommendation(user_profile, top_n=1)

        if user_profile.language == "spanish":
            st.markdown("### Según tus preferencias, te recomiendo las siguientes comidas:")
        elif user_profile.language == "portuguese":
            st.markdown("### Com base nas suas preferências, recomendo as seguintes refeições:")
        else:
            st.markdown("### Based on your preferences, I recommend the following meals:")

        for meal in recommendations:
            st.markdown(f"#### {meal['translated_title']}")
            if user_profile.language == "spanish" or user_profile.language == "portuguese":
                st.markdown("**Ingredientes:**")
            else:
                st.markdown("**Ingredients:**")
            
            st.markdown("\n".join([f"- {ingredient}" for ingredient in meal['translated_ingredients']]))
            if user_profile.language == "spanish":
                st.markdown("**Instrucciones:**")
            elif user_profile.language == "portuguese":
                st.markdown("**Instruções:**")
            else:
                st.markdown("**Directions:**")
            
            st.markdown("\n".join([f"{i+1}. {step}" for i, step in enumerate(meal['translated_directions'])]))

            spinner_message = f"Generating image for {meal['translated_title']}..."
            if user_profile.language == "spanish":
                spinner_message = f"Generando imagen para {meal['translated_title']}..."
            elif user_profile.language == "portuguese":
                spinner_message = f"Gerando imagem para {meal['translated_title']}..."
            with st.spinner(spinner_message):
                image_url = image_generator.generate_image(meal)

            if image_url:
                with st.spinner(spinner_message):
                    response = requests.get(image_url)
                    img = Image.open(BytesIO(response.content))
                if user_profile.language == "spanish":
                    st.image(img, caption=f"Imagen generada por IA de {meal['translated_title']}")
                elif user_profile.language == "portuguese":
                    st.image(img, caption=f"Imagem gerada por IA de {meal['translated_title']}")
                else:
                    st.image(img, caption=f"AI-generated image of {meal['translated_title']}")
            else:
                if user_profile.language == "spanish":
                    st.warning("No se pudo generar una imagen para esta comida.")
                elif user_profile.language == "portuguese":
                    st.warning("Não foi possível gerar uma imagem para esta refeição.")
                else:
                    st.warning("Could not generate an image for this meal.")
            st.markdown("---")

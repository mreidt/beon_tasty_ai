import logging
from io import BytesIO

import requests
import streamlit as st
from image_generator import ImageGenerator
from nlp import NLP
from PIL import Image
from recommendation import Recommendation
from vectorizer import Vectorizer

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

logger.debug("Starting TastyAI chat app")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

st.title("TastyAI")
with st.form("my_form"):
    text = st.text_area(
        "Enter text (we can answer in portuguese, spanish and english):",
        value=("I want a sugared meal that does not contain too much sugar and that I can share with my husband."),
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        with st.spinner("Working on it..."):
            nlp = NLP(openai_api_key=openai_api_key)

            user_profile = nlp.process_user_input(text)

        if not user_profile.is_recipe_request:
            if user_profile.language == "spanish":
                st.warning("¡Por favor, pide una receta!", icon="⚠")
            elif user_profile.language == "portuguese":
                st.warning("Por favor, peça uma receita!", icon="⚠")
            else:
                st.warning("Please ask for a recipe!", icon="⚠")
            st.stop()

        with st.spinner("Working on it..."):
            if user_profile.language == "spanish":
                initial_message = "Generando recomendaciones de comidas..."
            elif user_profile.language == "portuguese":
                initial_message = "Gerando recomendações de refeições..."
            else:
                initial_message = "Generating meal recommendations..."

        with st.spinner(initial_message):
            vectorizer = Vectorizer("./tastyai/src/dataset/full_dataset.csv")
            image_generator = ImageGenerator(openai_api_key)
            recommendation = Recommendation(vectorizer, openai_api_key)
            logger.debug("Getting recommendations...")
            recommendations = recommendation.get_meal_recommendation(user_profile, top_n=3)
            logger.debug(f"Recommendations: {recommendations}")

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

                st.markdown("\n".join([f"- {ingredient}" for ingredient in meal["translated_ingredients"]]))
                if user_profile.language == "spanish":
                    st.markdown("**Instrucciones:**")
                elif user_profile.language == "portuguese":
                    st.markdown("**Instruções:**")
                else:
                    st.markdown("**Directions:**")

                st.markdown("\n".join([f"{i + 1}. {step}" for i, step in enumerate(meal["translated_directions"])]))

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
                        st.image(
                            img,
                            caption=(f"Imagen generada por IA de {meal['translated_title']}"),
                        )
                    elif user_profile.language == "portuguese":
                        st.image(
                            img,
                            caption=(f"Imagem gerada por IA de {meal['translated_title']}"),
                        )
                    else:
                        st.image(
                            caption=(f"AI-generated image of {meal['translated_title']}"),
                        )
                else:
                    if user_profile.language == "spanish":
                        st.warning("No se pudo generar una imagen para esta comida.")
                    elif user_profile.language == "portuguese":
                        st.warning("Não foi possível gerar uma imagem para esta refeição.")
                    else:
                        st.warning("Could not generate an image for this meal.")
                st.markdown("---")

from vectorizer import Vectorizer
import logging
from nlp import NLP
from recommendation import Recommendation
from image_generator import ImageGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print(logger)

logger.info("Starting TastyAI chat app")
openai_api_key = ""
while not openai_api_key.startswith("sk-"):
    openai_api_key = input("Please enter your OpenAI API key, or quit to exit: ")
    if openai_api_key == "quit":
        print("Exiting...")
        exit()

if openai_api_key.startswith("sk-"):
    initial_chat_message = "Enter text (we can answer in portuguese, spanish and english). To exit say quit:"
    text = input(initial_chat_message)
    if text == "quit":
        print("Exiting...")
        exit()
    nlp = NLP(openai_api_key=openai_api_key)

    user_profile = nlp.process_user_input(text)
        
    while not user_profile.is_recipe_request:
        if user_profile.language == "spanish":
            print("¡Por favor, pide una receta!")
        elif user_profile.language == "portuguese":
            print("Por favor, peça uma receita!")
        else:
            print("Please ask for a recipe!")

        text = input(initial_chat_message)
        if text == "quit":
            print("Exiting...")
            exit()

    while text != "quit":
        if user_profile.language == "spanish":
            initial_message = "Generando recomendaciones de comidas..."
        elif user_profile.language == "portuguese":
            initial_message = "Gerando recomendações de refeições..."
        else:
            initial_message = "Generating meal recommendations..."

        print(initial_message)
        vectorizer = Vectorizer('./tastyai/src/dataset/full_dataset.csv')
        image_generator = ImageGenerator(openai_api_key)
        recommendation = Recommendation(vectorizer, openai_api_key)
        logger.debug("Getting recommendations...")
        recommendations = recommendation.get_meal_recommendation(user_profile, top_n=1)
        logger.debug(f"Recommendations: {recommendations}")

        if user_profile.language == "spanish":
            print("### Según tus preferencias, te recomiendo las siguientes comidas:")
        elif user_profile.language == "portuguese":
            print("### Com base nas suas preferências, recomendo as seguintes refeições:")
        else:
            print("### Based on your preferences, I recommend the following meals:")

        for meal in recommendations:
            print(f"#### {meal['translated_title']}")
            if user_profile.language == "spanish" or user_profile.language == "portuguese":
                print("**Ingredientes:**")
            else:
                print("**Ingredients:**")
                    
            print("\n".join([f"- {ingredient}" for ingredient in meal['translated_ingredients']]))
            if user_profile.language == "spanish":
                print("**Instrucciones:**")
            elif user_profile.language == "portuguese":
                print("**Instruções:**")
            else:
                print("**Directions:**")
                    
            print("\n".join([f"{i+1}. {step}" for i, step in enumerate(meal['translated_directions'])]))

            spinner_message = f"Generating image for {meal['translated_title']}..."
            if user_profile.language == "spanish":
                spinner_message = f"Generando imagen para {meal['translated_title']}..."
            elif user_profile.language == "portuguese":
                spinner_message = f"Gerando imagem para {meal['translated_title']}..."
            print(spinner_message)
            image_url = image_generator.generate_image(meal)

            if image_url:
                if user_profile.language == "spanish":
                    print(f"Imagen generada por IA de {meal['translated_title']}: {image_url}")
                elif user_profile.language == "portuguese":
                    print(f"Imagem gerada por IA de {meal['translated_title']}: {image_url}")
                else:
                    print(f"AI-generated image of {meal['translated_title']}: {image_url}")
            else:
                if user_profile.language == "spanish":
                    print("No se pudo generar una imagen para esta comida.")
                elif user_profile.language == "portuguese":
                    print("Não foi possível gerar uma imagem para esta refeição.")
                else:
                    print("Could not generate an image for this meal.")
            print("---")

        print("\n")
        text = input(initial_chat_message)
        if text == "quit":
            print("Exiting...")
            exit()

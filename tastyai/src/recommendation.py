import ast
import torch
import numpy as np
from vectorizer import Vectorizer
from user_profile import UserProfile
from translator import Translator
import torch.nn.functional as F

class Recommendation:
    def __init__(self, vectorizer: Vectorizer, openai_key):
        self.vectorizer = vectorizer
        self.embeddings = vectorizer.vectorize()
        self.recipes_df = vectorizer.df
        self.model = vectorizer.model
        self.openai_key = openai_key

        self.embeddings_tensor = torch.tensor(self.embeddings, dtype=torch.float32, device=self.vectorizer.device)
    
    def __string_to_list(self, s):
        """Convert a string representation of a list into an actual list."""
        try:
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return [item.strip() for item in s.strip('[]').split(',')]

    def get_meal_recommendation(self, user_profile: UserProfile, top_n=5):
        """Generate personalized meal recommendations."""
        query_string = " ".join(user_profile.dietary_preferences + 
                                user_profile.preferred_ingredients + 
                                [user_profile.sugar_preference])

        query_vector = self.model.encode([query_string], convert_to_tensor=True)
        similarity_scores = F.cosine_similarity(query_vector, self.embeddings_tensor).cpu().numpy()
        # Add small random noise to avoid ties
        similarity_scores += np.random.uniform(0, 0.01, size=len(similarity_scores))
        top_indices = similarity_scores.argsort()[-top_n:][::-1]

        recommendations = []
        for index in top_indices:
            suggested_meal = self.recipes_df.iloc[index].copy()

            if isinstance(suggested_meal["ingredients"], str):
                suggested_meal["ingredients"] = self.__string_to_list(suggested_meal["ingredients"])
            if isinstance(suggested_meal["directions"], str):
                suggested_meal["directions"] = self.__string_to_list(suggested_meal["directions"])

            recommendations.append(suggested_meal)
        
        return self.filter_recommendations(recommendations, user_profile)

    def filter_recommendations(self, recommendations, user_profile: UserProfile):
        """Apply additional filtering based on user preferences and optimize translations."""
        filtered_recommendations = []
        for meal in recommendations:
            if any(ingredient in meal["ingredients"] for ingredient in user_profile.excluded_ingredients):
                continue
            
            if user_profile.sugar_preference == "sugar_free" and any("sugar" in ingredient.lower() for ingredient in meal["ingredients"]):
                continue
            
            translated_meal = meal.copy()
            if user_profile.language == "english":
                translated_meal["translated_title"] = meal["title"]
                translated_meal["translated_ingredients"] = meal["ingredients"]
                translated_meal["translated_directions"] = meal["directions"]
            else:
                translator = Translator(openai_key=self.openai_key, language=user_profile.language)

                ingredients_text = " | ".join(meal["ingredients"])
                directions_text = " | ".join(meal["directions"])

                translated_meal["translated_title"] = translator.translate(meal["title"])
                translated_meal["translated_ingredients"] = translator.translate(ingredients_text).split(" | ")
                translated_meal["translated_directions"] = translator.translate(directions_text).split(" | ")

            filtered_recommendations.append(translated_meal)
            
        return filtered_recommendations


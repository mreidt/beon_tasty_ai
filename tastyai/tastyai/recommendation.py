import ast
from tastyai.vectorizer import Vectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tastyai.user_profile import UserProfile
from tastyai.translator import Translator

class Recommendation:
    def __init__(self, vectorizer: Vectorizer, openai_key):
        self.vectorizer = vectorizer
        self.X = vectorizer.vectorize()
        self.recipes_df = vectorizer.df
        self.openai_key = openai_key
    
    def __string_to_list(self, s):
        try:
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return [item.strip() for item in s.strip('[]').split(',')]

    def get_meal_recommendation(self, user_profile: UserProfile, top_n=5):
        query_string = " ".join(user_profile.dietary_preferences + 
                                user_profile.preferred_ingredients + 
                                [user_profile.sugar_preference])
        query_vector = self.vectorizer.vectorizer.transform([query_string])
        similarity_scores = cosine_similarity(query_vector, self.X).flatten()
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        
        recommendations = []
        for index in top_indices:
            suggested_meal = self.recipes_df.iloc[index]
            final_recommendation = suggested_meal.copy()
            
            if isinstance(final_recommendation["ingredients"], str):
                final_recommendation["ingredients"] = self.__string_to_list(final_recommendation["ingredients"])
            if isinstance(final_recommendation["directions"], str):
                final_recommendation["directions"] = self.__string_to_list(final_recommendation["directions"])
            
            recommendations.append(final_recommendation)
        
        return self.filter_recommendations(recommendations, user_profile)

    def filter_recommendations(self, recommendations, user_profile: UserProfile):
        filtered_recommendations = []
        for meal in recommendations:
            # Check for excluded ingredients
            if any(ingredient in meal["ingredients"] for ingredient in user_profile.excluded_ingredients):
                continue
            
            # Check sugar content
            if user_profile.sugar_preference == "sugar_free" and any("sugar" in ingredient.lower() for ingredient in meal["ingredients"]):
                continue
            
            translated_meal = meal.copy()
            if user_profile.language == "english":
                translated_meal["translated_title"] = meal["title"]
                translated_meal["translated_ingredients"] = meal["ingredients"]
                translated_meal["translated_directions"] = meal["directions"]
            else:
                translator = Translator(openai_key=self.openai_key, language=user_profile.language)
                translated_meal["translated_title"] = translator.translate(meal["title"])
                translated_meal["translated_ingredients"] = [translator.translate(ingredient) for ingredient in meal["ingredients"]]
                translated_meal["translated_directions"] = [translator.translate(step) for step in meal["directions"]]

            filtered_recommendations.append(translated_meal)
            
        return filtered_recommendations

class UserProfile:
    def __init__(self, preferences):
        self.dietary_preferences = preferences.get('dietary', [])
        self.sugar_preference = preferences.get('sugar_content', 'normal')
        self.sharing = preferences.get('sharing', False)
        self.preferred_ingredients = preferences.get('ingredients', [])
        self.excluded_ingredients = preferences.get('excluded_ingredients', [])
        self.language = preferences.get('language', 'english')
        self.is_recipe_request = preferences.get('is_recipe_request', False)

    def __str__(self):
        return (
            f"Dietary: {self.dietary_preferences}, Sugar: {self.sugar_preference}, Sharing: {self.sharing}, "
            f"Preferred Ingredients: {self.preferred_ingredients}, Excluded: {self.excluded_ingredients}, "
            f"Language: {self.language}, Is Recipe Request: {self.is_recipe_request}"
        )

    def __setattr__(self, name, value):
        if name == 'sugar_preference' and value not in ['low', 'normal', 'high', 'sugar_free']:
            raise ValueError("Invalid sugar preference. Choose from 'low', 'normal', 'high', or 'sugar_free'.")
        super().__setattr__(name, value)

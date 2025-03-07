# TastyAI

TastyAI is a smart, AI-powered meal recommendation platform that provides users with personalized meal suggestions based on their dietary preferences and constraints.

## Models and methodology

### User Input Processing

The system accepts user input in English, Spanish and Portuguese. Based on that, we translate the response according to the input language (this way, we don't need to translate to multiple languages every time we do a request).
for **Natural Language Processing (NLP)**, techniques are used to extract user preferences, dietary restrictions and recipe requirements.
The following models were used in this step:

| Model               |Why we choose this model                          |What is this model used for                         |
|----------------|-------------------------------|-----------------------------|
|*OpenAI gpt-4o-mini*| It's a small cost-efficient model, indicated for fast, everyday tasks            |Used for extracting user preferences from input queries   |

The user preference used for this project contains the following variables:

|Field| Type  | Explanation | Possible values
|--|--|--|--|
| Dietary preferences  | `List[str]` | A list of dietary preferences or restrictions the user has mentioned, such as "vegetarian," "vegan," "low-carb," etc | N/A |
| Sugar preference | `string` | Indicates the user's preference for sugar content in meals | `sugar_free`, `low`, `high`, `normal` (default) |
| Sharing | `bool` | Indicates whether the user intends to share the meal with others | `true`, `false` |
| Preferred ingredients | `List[str]` | A list of specific ingredients the user has expressed a preference for | N/A |
| Excluded ingredients | `List[str]` | A list of ingredients the user wants to avoid in their meals | N/A |
| Language | `string` | The language in which the user made their request | `english`, `spanish`, `portuguese` |
| Is recipe request | `bool` | Indicates whether the user's input is actually a request for a recipe recommendation | `true`, `false` |

If the system identifies that the user is not requesting a recipe, a warning message is sent to the user, and this user will need to resend the prompt.

### Data Vectorization

The vectorizer code expects the dataset to be put under the the `tastyai/tastyai/dataset` folder, with the name `full_dataset.csv`. First time the code runs, the system does the vectorization and stores it on the same folder, with the name `embeddings.npy`. Next time, vectorization code will use this stored file, so it doesn't need to do the process again.
To avoid high memory consumption, the file is loaded in chunks (default is 50.000). We also use torch `DataLoader`,  to process the data in batches (default size is 32).
After process the chunk, the file is incrementally saved to disk, avoiding RAM overload. The system also verifies if a GPU is available, in order to speed up the process. If no GPU is available, CPU is used for this.
The embeddings are loaded using memory map, to avoid high RAM consumption.
For the embeddings we are using the columns title, NER and ingredients. 
The following models were used in this step:

| Model               |Why we choose this model                          |What is this model used for                         |
|----------------|-------------------------------|-----------------------------|
|*all-MiniLM-L6-v2*| Efficiency and speed, versatility, quality of embeddings, resource efficiency and easy integration | Generating embeddings of recipe features |

### Recommendation Generation
The recommendation generation code transforms user preferences into a query vector. After that, it runs a cosine similarity between the query vector and the recipe embeddings, to find the most relevant matches. 
Since the user can have restrictions like sugar or only vegan food, additional filtering is applied, excluding ingredients and restricted dietary from the suggestions.
Also, in this part we do the translation of the recipes, but this part will be better explained in one specific topic.
The following models were used in this step:

| Model               |Why we choose this model                          |What is this model used for                         |
|----------------|-------------------------------|-----------------------------|
|*cosine similarity*| Efficiency and robustness | Compare the query vector with the pre-computed recipe embeddings |

### Translation
Translation uses *langchain* to access OpenAI and translates batches of texts from english to spanish or portuguese. The expected response is a string, with the translated text. **We only do the translation if the user input is NOT in english**.
To avoid multiple calls to the OpenAI API, the ingredients are combined using one specific separator. We do the same for directions. We then do 3 calls to the API: 1 for the title, 1 for the ingredients and 1 for the directions.
The result is splitted again, to show the ingredients and directions separated on the response.
The following models were used in this step:

| Model               |Why we choose this model                          |What is this model used for                         |
|----------------|-------------------------------|-----------------------------|
|*gpt-4o-mini*| It’s a small cost-efficient model, indicated for fast, everyday tasks | Translation from english to spanish and portuguese |

### Image generation
For image generation we use the OpenAI Python SDK. We set a prompt that contains the title and the ingredients of the recipe. The default values are:
| Field | Value |
|--|--|
| Samples | 1 |
| Size | 512x512 |

The following models were used in this step:

| Model               |Why we choose this model                          |What is this model used for                         |
|----------------|-------------------------------|-----------------------------|
|*dall-e-2*| It's cheap and generate good images  | Recipe image generation |

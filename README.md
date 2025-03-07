# beon_tasty_ai
TastyAI is a smart, AI-powered meal recommendation platform that provides users with personalized meal suggestions based on their dietary preferences and constraints.

You can find the methodology documentation [here](tastyai/doc/methodology.md).

Query and response examples can be found [here](tastyai/doc/examples.md).

## How to run the project

The first step to execute the project is to put the dataset inside `tastyai/src/dataset` folder. You should put the file with the name `full_dataset.csv`.

We have two types of execution for this project. The first option uses `streamlit`, so you can have an UI directly on your browser (simulates chat gpt interface). To run in this mode, go to the root of the project, then go to `tastyai` and then run:
- `make run_streamlit`

If you don't want an UI, just want to run on the terminal, you should go to the root of the project, then go to `tastyai` and then run:
- `make run_terminal`

Those commands will install all the required libraries and then run the project.

**NOTE**: *on the first time you run the project, it will vectorize the dataset. This can take **a lot of time** to execute.*

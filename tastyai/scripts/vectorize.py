from tastyai.src.vectorizer import Vectorizer

def run():
    print("Running vectorize.py...")
    vectorizer = Vectorizer('./tastyai/src/dataset/full_dataset.csv')
    _ = vectorizer.vectorize()
    print("Vectorization complete.")

if __name__ == "__main__":
    run()

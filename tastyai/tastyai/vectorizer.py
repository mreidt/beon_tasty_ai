import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from scipy import sparse
import os
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Vectorizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vectorized_file_path = Path('./tastyai/dataset/vectorized_data.npz')

    def __string_to_list(self, s):
        try:
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return []
    
    def __get_ner_column(self):
        if 'NER' not in self.df.columns:
            raise ValueError("The dataset does not have a 'NER' column.")
        df = self.df.copy()
        df = df[["NER"]]
        df.loc[:, 'NER'] = df["NER"].apply(self.__string_to_list)
        df.loc[:, 'NER_string'] = df["NER"].apply(lambda x: " ".join(x))
        logger.info(f"df columns: {df.head(5)}")
        return df
    
    def vectorize(self):
        if self.vectorized_file_path.exists():
            logger.info("Loading existing vectorized data...")
            X = sparse.load_npz(self.vectorized_file_path)
            self.vectorizer = joblib.load(self.vectorized_file_path.with_suffix('.joblib'))
            return X
        
        logger.info("Vectorizing data...")
        df = self.__get_ner_column()
        logger.info(f"df columns: {df.columns}")
        X = self.vectorizer.fit_transform(df["NER_string"])
        
        os.makedirs(os.path.dirname(self.vectorized_file_path), exist_ok=True)
        
        sparse.save_npz(self.vectorized_file_path, X)
        joblib.dump(self.vectorizer, self.vectorized_file_path.with_suffix('.joblib'))
        logger.info(f"Vectorized data saved to {self.vectorized_file_path}")
        
        return X

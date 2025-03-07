import ast
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Vectorizer:
    def __init__(self, file_path, batch_size=32, use_gpu=True, chunk_size=50000):
        self.file_path = file_path
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        logger.debug(f"Using device: {self.device}")
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        self.embeddings_file_path = Path(self.file_path).with_name("embeddings.npy")
        self.metadata_file_path = Path(self.file_path).with_name("metadata.pkl")
        logger.debug(f"Embeddings file path: {self.embeddings_file_path}")
        logger.debug(f"Metadata file path: {self.metadata_file_path}")

        self.df = pd.read_csv(self.file_path)

    def __string_to_list(self, s):
        """Convert a string representation of a list into an actual list."""
        try:
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return []

    def __get_combined_features(self, df):
        """Combine title, ingredients, and named entities (NER) into a single text feature."""
        df["NER"] = df["NER"].apply(self.__string_to_list)
        df["ingredients"] = df["ingredients"].apply(self.__string_to_list)

        return (
            df["title"]
            + " "
            + df["NER"].apply(lambda x: " ".join(x))
            + " "
            + df["ingredients"].apply(lambda x: " ".join(x))
        ).tolist()

    def vectorize(self):
        """Efficiently process and vectorize dataset in chunks without loading everything into memory."""
        if self.embeddings_file_path.exists():
            logger.debug("Loading existing embeddings from disk...")
            return np.load(self.embeddings_file_path, mmap_mode="r")

        logger.debug(f"Processing dataset in chunks (Chunk Size: {self.chunk_size})...")

        start_time = time.time()

        first_pass = True
        total_rows = 0
        embeddings_list = []

        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            chunk_start_time = time.time()
            combined_texts = self.__get_combined_features(chunk)

            data_loader = DataLoader(combined_texts, batch_size=self.batch_size, shuffle=False)

            chunk_embeddings = []
            for batch in data_loader:
                batch_embeddings = self.model.encode(batch, convert_to_tensor=True, device=self.device)
                chunk_embeddings.append(batch_embeddings.cpu().numpy())

            chunk_embeddings = np.vstack(chunk_embeddings)
            embeddings_list.append(chunk_embeddings)
            total_rows += len(chunk)

            logger.debug(f"Processed {total_rows} recipes... (Chunk Time: {time.time() - chunk_start_time:.2f}s)")

            if first_pass:
                np.save(self.embeddings_file_path, chunk_embeddings)
                first_pass = False
            else:
                existing_data = np.load(self.embeddings_file_path, mmap_mode="r")
                updated_data = np.vstack([existing_data, chunk_embeddings])
                np.save(self.embeddings_file_path, updated_data)

        total_time = time.time() - start_time
        logger.debug(f"Total embedding time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

        pd.to_pickle({"num_rows": total_rows, "time_taken": total_time}, self.metadata_file_path)

        return np.load(self.embeddings_file_path, mmap_mode="r")

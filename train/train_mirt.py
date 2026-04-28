import argparse
import logging
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from MIRT import MIRT
import torch
import pickle
from pathlib import Path

ROOT = Path(__file__).parent.parent  # llm_routing/
MODELS_DIR = ROOT / "results" / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_embeddings(embeddings_dir):
    with open(ROOT / f"utils/{embeddings_dir}_embeddings/llm_embeddings.pkl", "rb") as f:
        llm_embeddings_data = pickle.load(f)
    llm_embeddings = {llm["index"]: np.array(llm["embedding"]) for llm in llm_embeddings_data}
    with open(ROOT / f"utils/{embeddings_dir}_embeddings/query_embeddings.pkl", "rb") as f:
        query_embeddings_data = pickle.load(f)
    query_embeddings = {query["index"]: np.array(query["embedding"]) for query in query_embeddings_data}

    llm_id_map = pd.read_csv(ROOT / "utils/map/llm.csv", index_col="name").to_dict()["index"]
    query_id_map = pd.read_csv(ROOT / "utils/map/query.csv", index_col="question").to_dict()["index"]
    return llm_embeddings, query_embeddings, llm_id_map, query_id_map


def map_ids_to_vectors(data, llm_embeddings, query_embeddings, llm_id_map, query_id_map):
    llm_vectors = []
    query_vectors = []
    for _, row in data.iterrows():
        llm_id = llm_id_map[row["llm"]]
        query_id = query_id_map[row['question']]
        llm_vectors.append(llm_embeddings[llm_id])
        query_vectors.append(query_embeddings[query_id])
    return np.array(llm_vectors), np.array(query_vectors)


def get_embedding_dims(emb_name):
    """
    Get embedding dimensions for a given embedding type.
    
    Parameters
    ----------
    emb_name : str
        Embedding type ('open', 'zhipu', 'bge', or 'bert')
    
    Returns
    -------
    tuple
        (llm_dim, query_dim)
    """
    if emb_name == "open":
        return 1536, 1536
    elif emb_name == "zhipu":
        return 512, 512
    elif emb_name == "bge":
        return 1024, 1024
    elif emb_name == "bert":
        return 768, 768
    else:
        raise ValueError(f"Unknown embedding type: {emb_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", type=str, default="bert", choices=["bert", "open", "zhipu", "bge"])
    args = parser.parse_args()
    emb_name = args.embedding
    
    train_data = pd.read_csv(ROOT / "data/irt_data/train.csv")
    test_data = pd.read_csv(ROOT / "data/irt_data/test1.csv")

    llm_embeddings, query_embeddings, llm_id_map, query_id_map = load_embeddings(emb_name)

    train_llm, train_query = map_ids_to_vectors(train_data, llm_embeddings, query_embeddings, llm_id_map, query_id_map)
    test_llm, test_query = map_ids_to_vectors(test_data, llm_embeddings, query_embeddings, llm_id_map, query_id_map)

    batch_size = 512
    train_set = DataLoader(TensorDataset(
        torch.tensor(train_llm, dtype=torch.float32),
        torch.tensor(train_query, dtype=torch.float32),
        torch.tensor(train_data["performance"].values, dtype=torch.float32)
    ), batch_size=batch_size, shuffle=True)

    test_set = DataLoader(TensorDataset(
        torch.tensor(test_llm, dtype=torch.float32),
        torch.tensor(test_query, dtype=torch.float32),
        torch.tensor(test_data["performance"].values, dtype=torch.float32)
    ), batch_size=batch_size, shuffle=False)

    llm_dim, query_dim = get_embedding_dims(emb_name)
    knowledge_n = 25
        

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info("Using device: %s", device)

    cdm = MIRT(llm_dim, query_dim, knowledge_n)
    cdm.train(train_set, test_set, epoch=9, device=device)
    cdm.save(MODELS_DIR / f"mirt_{emb_name}.snapshot")
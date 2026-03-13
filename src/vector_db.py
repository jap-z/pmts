import faiss
import numpy as np
import os
import argparse
from tqdm import tqdm

class VectorDatabase:
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        # IndexFlatIP + L2 normalization = Cosine Similarity
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata = None # Will store timestamps and y_labels

    def load_data(self, data_path='data/btc_6h_embeddings.npz'):
        print(f"Loading embeddings from {data_path}...")
        data = np.load(data_path)
        
        # We need: embeddings, timestamps, and the truth labels
        embeddings = data['embeddings'].astype('float32')
        
        # L2 Normalize for Cosine Similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        print(f"Indexed {self.index.ntotal} vectors.")
        
        self.metadata = {
            'timestamps': data['timestamps'],
            'y_close_ret': data['y_close_ret'],
            'y_max_ret': data['y_max_ret'],
            'y_min_ret': data['y_min_ret']
        }
        return self.index.ntotal

    def query(self, query_vector, k=10):
        """
        query_vector: 1D or 2D array of shape (128,) or (N, 128)
        """
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        query_vector = query_vector.astype('float32')
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            results.append({
                'index': int(idx),
                'score': float(distances[0][i]),
                'timestamp': str(self.metadata['timestamps'][idx]),
                'next_24h_return': float(self.metadata['y_close_ret'][idx]),
                'next_24h_max': float(self.metadata['y_max_ret'][idx]),
                'next_24h_min': float(self.metadata['y_min_ret'][idx])
            })
        return results

if __name__ == "__main__":
    vdb = VectorDatabase()
    vdb.load_data()
    
    # Test query: Pick a random sample from the index itself
    test_idx = np.random.randint(0, vdb.index.ntotal)
    data = np.load('data/btc_6h_embeddings.npz')
    test_vec = data['embeddings'][test_idx]
    
    print(f"\n--- Testing Similarity Search ---")
    print(f"Querying for pattern at: {data['timestamps'][test_idx]}")
    
    matches = vdb.query(test_vec, k=5)
    
    for i, m in enumerate(matches):
        print(f"Match {i+1}: {m['timestamp']} | Similarity: {m['score']:.4f} | Next 24h: {m['next_24h_return'] * 100:.2f}%")

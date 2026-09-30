import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error

def extract_latent_embeddings(model, dataloader, device):
    """Passes data through Autoencoder to extract latent space embeddings."""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            _, latent = model(X_batch)
            embeddings.append(latent.cpu().numpy())
            
    return np.vstack(embeddings)

def cluster_regimes(embeddings: np.ndarray, n_clusters: int = 4):
    """Clusters the latent embeddings into distinct Market Regimes."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Calculate Silhouette score (only if valid)
    if 1 < n_clusters < len(embeddings):
        sil_score = silhouette_score(embeddings, labels)
    else:
        sil_score = -1.0
        
    return labels, kmeans, sil_score

def evaluate_lstm(model, dataloader, device):
    """Evaluates the LSTM predictor on MSE and Directional Accuracy."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            
    preds = np.vstack(all_preds).flatten()
    targets = np.vstack(all_targets).flatten()
    
    mse = mean_squared_error(targets, preds)
    
    # Directional Accuracy (did they both move in the same direction?)
    # Sign of return: >0 is 1, <=0 is 0
    pred_dir = (preds > 0).astype(int)
    target_dir = (targets > 0).astype(int)
    
    acc = np.mean(pred_dir == target_dir)
    
    return mse, acc

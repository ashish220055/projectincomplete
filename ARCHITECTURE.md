# Market Microstructure Pattern Discovery Platform
**Project Architecture Document (Deep Learning Edition)**

---

## 1. Problem Statement

* **What is Market Microstructure?** It is the study of how financial markets operate at a highly granular level, focusing on the mechanics of trading, order flow, liquidity, and how price discovery happens on a minute-by-minute or tick-by-tick basis.
* **What Problem This Solves:** Financial markets constantly shift between different "regimes" or "states". Most trading algorithms fail when the regime changes. By using deep learning to identify underlying latent market states, we can adapt strategies dynamically.
* **Why Pattern Discovery Over Price Prediction:** Predicting the exact future price of an asset is notoriously difficult due to extreme noise. Pattern discovery via Autoencoders focuses on finding structural, latent representations in market conditions. Knowing that "we are currently in a high-volatility, low-liquidity state" is far more robust and actionable for risk management than guessing the exact price movement.

---

## 2. Scope of Version 1

This version focuses on building a robust Deep Learning pipeline for advanced financial time-series modeling.

* **Language:** Python
* **Libraries Allowed:** `pytorch`, `pandas`, `numpy`, `matplotlib`, `scikit-learn` (for scaling/metrics)
* **Included:** Static data ingestion, advanced feature engineering (time-series windowing), **Autoencoders** for latent pattern discovery (unsupervised), **LSTMs** for price prediction, and PyTorch training loops.
* **Excluded:** Distributed systems, cloud deployment (AWS/GCP), real-time trading (websockets), and high-frequency tick data (MVP uses 1m klines).

---

## 3. Final Deliverables

Upon completion, this project will produce:

1. **Feature Sequences:** A preprocessed tensor dataset shaped for sequence modeling `(batch, sequence_length, features)`.
2. **Trained Models:** Saved PyTorch weights (`.pt` or `.pth`) for both the Autoencoder and LSTM models.
3. **Latent Embeddings:** Extracted latent representations of the market microstructure.
4. **Regime Discovery:** Clustering applied on top of Autoencoder embeddings to identify distinct market regimes.
5. **Evaluation Report:** Metrics table for both Autoencoder reconstruction loss and LSTM predictive performance.
6. **End-to-End Pipeline:** A demo script to run inference on new data.

---

## 4. Recommended Dataset

| Attribute | Specification |
| :--- | :--- |
| **Source** | Binance Vision (Historical Data) or Kaggle (Binance 1m data) |
| **Asset** | BTC/USDT (Bitcoin to Tether) |
| **Frequency** | 1-minute intervals (klines/candlesticks) |
| **Required Columns** | `timestamp`, `open`, `high`, `low`, `close`, `volume`, `number_of_trades`, `taker_buy_base_asset_volume` |
| **Format** | CSV |
| **Approx. Size** | 1 Year of Data (~525,600 rows, approx. 50MB) |

---

## 5. System Workflow

```text
Raw Data (CSV) 
      ↓
[ Data Loader & Cleaner ] → Cleaned DataFrame
      ↓
[ Feature Generator ] → Engineered DataFrame
      ↓
[ Sequence Windowing (PyTorch Dataset) ] → Tensors (B, S, F)
      ↓
      ├──→ [ Autoencoder ] → Reconstruction Loss & Latent Embeddings (Market States)
      │
      └──→ [ LSTM Predictor ] → MSE Loss & Price/Return Predictions
                  ↓
           [ Evaluation & Clustering ] → Silhouette (Embeddings) & Prediction Metrics
                  ↓
   Visual Plots & Final Evaluation Report
```

---

## 6. Project Folder Structure

```text
market_microstructure/
│
├── data/
│   ├── raw/               
│   └── processed/         
│
├── notebooks/
│   └── 01_model_exploration.ipynb  
│
├── src/
│   ├── __init__.py
│   ├── config.py          # Global parameters (sequence length, epochs, batch size)
│   ├── data_pipeline.py   # Loader, cleaner, and feature generator
│   ├── dataset.py         # PyTorch Dataset/DataLoader for time-series windowing
│   ├── models/            
│   │   ├── autoencoder.py # Autoencoder architecture
│   │   └── lstm.py        # LSTM architecture
│   ├── trainer.py         # PyTorch training loops (train, validate, test)
│   └── evaluation.py      # Metric calculations and plotting
│
├── models/
│   └── weights/           # Saved PyTorch .pth files
│
├── reports/
│   └── metrics.txt        
├── requirements.txt       
└── README.md              
```

---

## 7. Module Breakdown

| Module | Objective | Inputs | Outputs | Dependencies | Est. Effort |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Data Pipeline** | Load CSV, clean, generate features, and scale | Raw CSV | Cleaned & Scaled DataFrame | `pandas`, `scikit-learn` | Low (1-2 days) |
| **PyTorch Dataset** | Convert DataFrame to rolling window Tensors | DataFrame | PyTorch DataLoaders | `torch`, `numpy` | Low (1 day) |
| **Autoencoder Model**| Encode sequences into latent states | DataLoader (Tensors)| Reconstructed Tensors & Embeddings| `torch` | Med (2 days) |
| **LSTM Predictor** | Predict next time step return | DataLoader (Tensors)| Prediction Tensors | `torch` | Med (2 days) |
| **Trainer Engine** | Handle forward/backward passes and epochs | Models, DataLoaders | Saved Model Weights | `torch` | High (3 days) |
| **Evaluation** | Compute metrics, cluster embeddings, plot results| Embeddings, Predictions| Metrics Dict, Plots | `scikit-learn`, `matplotlib` | Med (2 days) |
| **API Backend** | Serve models via REST endpoints | JSON Requests | JSON Responses | `fastapi`, `uvicorn` | Low (1 day) |

---

## 8. Development Order

1. **Config & Environment:** Set up PyTorch and folder structure.
2. **Data Pipeline & Dataset:** Implement preprocessing and the crucial PyTorch sequence windowing.
3. **Autoencoder Network:** Build the architecture and training loop for unsupervised learning.
4. **LSTM Predictor Network:** Build the supervised architecture and training loop.
5. **Trainer Engine:** Standardize the training loop with early stopping.
6. **Evaluation & Clustering:** Cluster the extracted embeddings from the Autoencoder to label market regimes.
7. **Visualization:** Plot the regimes and predictions.

---

## 9. Git Workflow

* **Main Branch (`main`):** Stable, working code.
* **Feature Branches (`feature/<module-name>`):** Work on one module at a time (e.g., `feature/autoencoder`).
* **Commit Conventions:**
  * `feat:` for new features (e.g., `feat: implement LSTM architecture`)
  * `fix:` for bug fixes (e.g., `fix: resolve tensor shape mismatch in DataLoader`)
  * `train:` for model training updates/config changes

---

## 10. Chat-Isolated Development Strategy

**Copy-Paste Template for New Chats:**
```text
I am building a Deep Learning Market Microstructure Platform in PyTorch.
We are working ONLY on Module: [INSERT MODULE NAME, e.g., Autoencoder Model].

Here is the context:
- Project uses PyTorch, pandas, numpy.
- The input to this module is: [INSERT INPUT, e.g., a PyTorch DataLoader yielding Tensors of shape (Batch, Seq_Len, Features)]
- The expected output of this module is: [INSERT OUTPUT, e.g., Reconstructed Tensors and Latent Embeddings]

Task: Write the complete implementation code for `src/[filename].py` with PyTorch best practices. Ensure tensors are properly moved to device (CPU/GPU). Do not write code for other modules.
```

---

## 11. Interface Contracts

| Module Function | Input Schema | Output Schema |
| :--- | :--- | :--- |
| `process_data(path)` | `str` (valid filepath) | `DataFrame` (scaled, feature-rich) |
| `create_dataloaders(df, seq_len)` | `DataFrame`, `int` | `DataLoader` (Train), `DataLoader` (Test) |
| `Autoencoder.forward(x)` | `Tensor` (B, S, F) | `Tensor` (B, S, F) [Reconstruction], `Tensor` (B, Latent_Dim) [Embedding] |
| `LSTM.forward(x)` | `Tensor` (B, S, F) | `Tensor` (B, 1) [Prediction] |
| `train_model(...)` | `nn.Module`, `DataLoader`, `optimizer` | Saves `.pth` to disk, returns loss history |

---

## 12. Evaluation Plan

| Dimension | Metric | Success Criteria / Goal |
| :--- | :--- | :--- |
| **Autoencoder Quality** | Mean Squared Error (MSE) of Reconstruction | Low stable validation loss without overfitting |
| **Regime Quality** | Silhouette Score on Embeddings (via K-Means) | Score > 0.15 on the latent space |
| **Prediction Accuracy**| Directional Accuracy & MSE | Directional Accuracy > 51% |
| **Regime Stability** | Average Regime Duration | Regimes should intuitively group high/low volatility periods |

---

## 13. Risk Register

| Risk | Description | Mitigation Strategy |
| :--- | :--- | :--- |
| **Tensor Shape Mismatches**| `(Batch, Seq, Features)` vs `(Batch, Features, Seq)` errors in PyTorch. | Explicitly document `batch_first=True` in all RNN/LSTM modules. |
| **Look-Ahead Bias** | Scaling or windowing using future target data. | Fit `StandardScaler` strictly on the train split before creating sequence windows. |
| **Overfitting (Deep Learning)**| LSTMs memorizing noise instead of signal. | Implement robust Early Stopping, Dropout, and Weight Decay in the `trainer.py`. |
| **Non-Stationarity** | Financial data changes over time, breaking models. | Use stationary features (returns) rather than raw prices as inputs to the networks. |

---

## 14. Documentation Standards

* **README:** Instructions for setting up PyTorch (CUDA if applicable) and running training scripts.
* **Docstrings:** All PyTorch `nn.Module` classes must specify expected input and output tensor shapes in their docstrings.
* **Experiment Logs:** Training losses (train/val) per epoch should be logged to track overfitting.

---

## 15. Week-by-Week Roadmap

* **Week 1 (Data & Setup):** Setup PyTorch, implement `data_pipeline.py` and `dataset.py`. Ensure rolling windows are generated correctly.
* **Week 2 (Autoencoder):** Implement and train the Autoencoder. Extract embeddings and cluster them to define market regimes.
* **Week 3 (LSTM):** Implement and train the LSTM predictor. 
* **Week 4 (Evaluation):** Implement evaluation metrics, visualize latent regimes on price charts, and finalize the orchestration script.

---

## 16. Future Extensions (NOT V1)

Once the DL pipeline is functional, you can expand in these directions:
* **Transformers (Attention):** Replace LSTMs with Time-Series Transformers (e.g., Informer or Autoformer).
* **High-Frequency Tick Data:** Ingest Level 2 Order Book data and limit order flow instead of minute-level aggregates.
* **Live Streaming Inference:** Implement a websocket consumer for real-time model predictions.

---

## 17. Project Blueprint Summary

**Name:** Deep Learning Market Microstructure Platform
**Objective:** Discover latent market regimes using Autoencoders and predict baseline price movements using LSTMs on 1-minute cryptocurrency data.
**Core Tech:** PyTorch, Pandas, Scikit-Learn
**Data:** 1 Year BTC/USDT 1m Klines
**Workflow:** `Load/Clean -> Sequence Windowing -> Autoencoder (Embeddings) & LSTM (Predictions) -> Evaluate`
**Strict Constraints:** No look-ahead bias in sequence windowing; strictly monitor train/val loss for overfitting.
**Success Definition:** A clean PyTorch codebase that successfully trains an Autoencoder to extract meaningful market states (verified by clustering the embeddings) and an LSTM that provides baseline predictive utility.

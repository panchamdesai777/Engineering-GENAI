# Engineering-GENAI

This repository contains two tasks focused on the use of Machine Learning and Generative AI to extract insights from SEC EDGAR 10-K filings. The tasks aim to provide document understanding via clustering and extract structured financial insights using LLM-based retrieval.

## 📁 Dataset

- **Source**: [EDGAR Corpus on HuggingFace](https://huggingface.co/datasets/eloukas/edgar-corpus)
- **Filing Type**: 10-K
- **Years Covered**: 2018–2020
- **Documents Used**:
  - Engineering #1: 10 Companies (2020 only)
  - Gen AI #2: 1 Company (2018–2020)

---

## 🧠 Engineering #1 – Document Embedding & Clustering

**Objective**: Visualize and identify outlier document sections in 10-K filings using clustering techniques.

### ✅ Steps Implemented

1. **Chunking**  
   Documents were split into 500-word chunks by section.

2. **Embedding**  
   Each chunk was embedded using `Word2Vec`.

3. **Standardization**  
   Embeddings were scaled using `StandardScaler`.

4. **Dimensionality Reduction**  
   PCA followed by 2D projection for visualization.

5. **KMeans Clustering**  
   Clusters assigned to each chunk and outliers flagged based on distance from centroid.

6. **Visualization**  
   Generated 2D scatter plots colored by:
   - Cluster assignment
   - Outlier flag
   - Section wise

---

## 🤖 Gen AI #2 – Generative AI for Data Extraction

**Objective**: Extract structured insights from 10-K filings of a single company using LLM embeddings and similarity search.

### ✅ Steps Implemented

1. **Chunking**  
   Filings were segmented into section-based chunks.

2. **Embedding**  
   Chunks and queries were embedded using `SentenceTransformer`.

3. **Query Prompt**  
   Custom prompts were used to extract specific business information (e.g., company overview, risk factors, etc.).

4. **Similarity Retrieval**  
   Used a RAG-based approach with a FAISS vector database to retrieve the top relevant chunks.

5. **Validation**  
   Ground truth values were collected for 5 attributes to evaluate LLM extraction accuracy.

### 📌 Sample Attributes Extracted

- Business description
- Risk factors
- Dividends payment
- Financial overview
- Legal proceedings

---

## 🛠️ Tech Stack

- **Language**: Python
- **Frameworks**: PySpark, Hugging Face Transformers, Scikit-learn, Matplotlib , Langchain
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensionality Reduction**: PCA
- **Clustering**: KMeans
- **LLM** - GROQ API

---

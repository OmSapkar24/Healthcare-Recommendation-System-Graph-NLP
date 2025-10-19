# Advanced Recommendation System for Healthcare (NLP + Graph)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-Graph%20Learning-orange)](https://pyg.org/)
[![Transformers](https://img.shields.io/badge/HF-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-informational)](https://www.docker.com/)

Personalized care recommendations using:
- Clinical note understanding (BioClinicalBERT)
- Prescription graph learning (medication co-prescription, contraindications)
- Patient similarity graph with temporal features

## Overview
- NLP: encode clinical notes and diagnoses with domain LMs
- Graph: build bipartite graphs (patient–drug, patient–diagnosis); train GNN (GraphSAGE/GAT)
- Hybrid ranker: combine text similarity, GNN scores, rule constraints (age, allergies)
- Explainability: SHAP on text, GNN attention, counterfactuals

## Business Context
- Reduce adverse drug events and improve guideline adherence
- Support clinicians with transparent recommendations
- KPI targets: MAP@10 > 0.45, NDCG@10 > 0.55, ADE alerts recall > 0.9

## Tech Stack
- NLP: Transformers, sentence-transformers, scispaCy
- Graph: PyTorch Geometric, NetworkX, Neo4j optional
- Serving: FastAPI, Uvicorn, Docker
- MLOps: MLflow, DVC, GitHub Actions

## Repository Structure
```
healthcare-recsys/
  data/
  notebooks/
  src/
    nlp/
      encode_notes.py
      diagnosis_encoder.py
    graph/
      build_graph.py
      gnn_model.py
      candidate_gen.py
    rank/
      hybrid_ranker.py
    api/
      main.py
  configs/
  tests/
  docker/
```

## Sample Results (synthetic)
- MAP@10: 0.48
- NDCG@10: 0.58
- ADE recall: 0.92 at 0.1 FPR

## Installation
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_sci_lg
```

## Usage
Build graphs and train:
```
python -m src.graph.build_graph --ehr data/ehr.jsonl --rx data/prescriptions.csv
python -m src.graph.gnn_model --config configs/gnn.yaml
```
Start API:
```
uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

## Roadmap
- Temporal graph attention (TGN)
- Safety constraints via knowledge graphs (UMLS, RxNorm)
- Fairness evaluation across demographics

# Transparent Legal AI using Enhanced Conflict-Aware Retrieval and Re-ranking

This project is a non-legal, general-purpose research implementation of a transparent conflict-aware retrieval-augmented generation (RAG) system. It is designed to compare a baseline retrieval pipeline against a proposed conflict-aware system while remaining local-first and reproducible.

## Project goal

The system implements:

- semantic retrieval using ChromaDB and sentence embeddings
- semantic duplicate detection
- source credibility scoring
- recency-aware scoring
- contradiction detection
- conflict-aware re-ranking
- evidence selection
- local LLM generation with Ollama
- confidence estimation
- citation traceability
- experiments and ablation comparison

## Environment

Detected environment during setup:

- OS: Microsoft Windows 11 Home Single Language
- Python: 3.13.7
- Virtual environment: configured in the project root
- RAM: approximately 14 GB
- Disk: available local storage was checked and is sufficient for local development
- Git: installed
- Ollama: not installed in the current environment at initial inspection

## Setup

1. Create or activate the project virtual environment.
2. Install dependencies:

```powershell
cd "C:\Users\indud\OneDrive\Desktop\A1Major"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Install Ollama for Windows if needed:

```powershell
winget install Ollama.Ollama
```

Then verify the installation:

```powershell
ollama --version
```

4. Pull a practical local model:

```powershell
ollama pull llama3.2
```

5. Create the evaluation dataset and vector store:

```powershell
.\.venv\Scripts\python.exe -m scripts.ingest_data
```

## Run baseline

```powershell
.\.venv\Scripts\python.exe scripts/run_baseline.py
```

## Run proposed system

```powershell
.\.venv\Scripts\python.exe scripts/run_proposed.py
```

## Run experiments

```powershell
.\.venv\Scripts\python.exe scripts/run_experiments.py
```

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Launch Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Project structure

- app.py
- config/
- data/
- documentation in docs/
- modules/
- scripts/
- tests/
- results/

## Notes

- The project title is retained as required, but the actual implementation is general-purpose and non-legal.
- All examples and evaluation data use general research topics rather than legal content.
- This implementation uses a local, reproducible synthetic evaluation dataset plus real local embedding and retrieval stacks.

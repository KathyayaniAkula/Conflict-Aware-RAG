# Experimental Setup

The project includes a controlled synthetic evaluation dataset intended for general research experimentation.

## Dataset

The dataset is stored under data/evaluation and is explicitly labeled as controlled evaluation data. It includes:

- relevant and irrelevant documents
- exact and near duplicate content
- high and low credibility sources
- recent and older publications
- contradictory evidence

## Reproducibility

The system records key settings in config/config.py. These include:

- embedding model
- NLI model
- Ollama model
- top-K
- chunk size and overlap
- duplicate threshold
- contradiction threshold
- reranking weights
- evidence count
- random seed

## Running experiments

Use the scripts/run_experiments.py entry point to execute the ablation workflow.

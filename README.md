cat > ~/amr-sentinel/README.md << 'MDEOF'
# 🦠 AMR Sentinel

An agentic AI assistant for European Antimicrobial Resistance (AMR) surveillance — grounded in real EARS-Net/EFSA data, supporting One Health queries across human, animal, and food chain domains.

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red) ![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-green)

## Why AMR Sentinel?

| Problem with general LLMs | How AMR Sentinel solves it |
|---|---|
| Knowledge cutoff — resistance rates change every year | Queries live SQLite DB loaded from EARS-Net/EFSA data |
| Confidently hallucinate resistance percentages | Only cites [DATA] when rows actually exist in the database |
| Generic advice regardless of resistance levels | Recommendations triggered by real thresholds (CRITICAL ≥50%, HIGH ≥25%) |
| No One Health perspective | Integrates human clinical, animal, and food chain surveillance data |
| Cannot search recent literature | Semantic search over indexed PubMed abstracts via ChromaDB |

## Features

- **Natural language querying** — ask questions in plain English, get grounded answers
- **One Health data** — human bloodstream infections (ECDC EARS-Net) + animal/food chain resistance (EFSA)
- **Automated data ingestion** — downloads and parses ECDC annual PDF reports and EFSA Excel files
- **Semantic literature search** — PubMed abstracts embedded with all-MiniLM-L6-v2
- **Actionable recommendations** — clinical, public health, veterinary, and policy actions
- **Interactive Plotly dashboard** — trend lines, country comparisons, bubble charts, KPI metrics
- **Severity post-processing** — Python-guaranteed 🔴 CRITICAL / 🟠 HIGH / 🟡 MODERATE / 🟢 LOW labels
- **REST API** — FastAPI /ask endpoint for programmatic access
- **Provider-agnostic** — works with any OpenAI-compatible API (Ollama cloud, Anthropic, local)

## Architecture

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
User question
|
v
orchestrator.ask()
|
|-> Intent model (gemma3:4b)  ->  parse intent + extract params as JSON
|                                  (organism, antibiotic, country, year)
|
|-> data_agent    ->  SQL query on SQLite
|                     (trend / compare / top_resistant / list_values)
|
|-> lit_agent     ->  ChromaDB cosine search
|                     (top-k PubMed abstracts)
|
|-> Narrative model (gemma3:27b)  ->  synthesise grounded narrative
|                                      + severity post-processing (Python)
v
Streamlit UI / FastAPI response

## Data Sources

| Source | Coverage | Type |
|---|---|---|
| ECDC EARS-Net (PDF) | 2020–2024, 8 organisms, 30+ countries | Human bloodstream infections |
| EFSA Zoonotic (Excel) | 2015, E. coli + Salmonella, 31 countries | Animal (pigs) + food chain (pork meat) |
| PubMed via NCBI | ~200 abstracts | Scientific literature |

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Aamna-Khan-Git/amr-sentinel.git
cd amr-sentinel
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
nano .env
```

### 3. Ingest surveillance data

```bash
python3 pdf_ingest.py          # ECDC annual report PDF -> SQLite + ChromaDB
python3 fetch_efsa.py          # EFSA animal/food chain data -> SQLite
python3 fetch_abstracts_ncbi.py          # PubMed abstracts -> data/abstracts.jsonl
python3 literature_store.py data/abstracts.jsonl   # embed into ChromaDB
```

### 4. Launch

```bash
# Streamlit dashboard
streamlit run api_server.py

# REST API
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_KEY` | Yes | API key for your LLM provider |
| `API_BASE_URL` | Yes | LLM provider base URL |
| `INTENT_MODEL` | Yes | Model for intent parsing (fast/small) |
| `NARRATIVE_MODEL` | Yes | Model for narrative generation (quality) |
| `NCBI_API_KEY` | No | NCBI key (higher rate limits) |
| `NCBI_EMAIL` | No | Email for NCBI E-utilities |
| `DATABASE_PATH` | No | SQLite database path (default: amr_sentinel.db) |
| `CHROMA_PATH` | No | ChromaDB store path (default: chroma_store) |
| `PORT` | No | API server port (default: 8000) |

## Example Queries

| Question | Intent |
|---|---|
| "How has ciprofloxacin resistance in E. coli changed over time?" | trend |
| "Compare tetracycline resistance in Salmonella across countries" | compare |
| "I am from Germany, which antibiotics should I avoid?" | compare |
| "Which meat is safest in Bulgaria?" | compare |
| "Which organism/antibiotic combinations have the highest resistance?" | top_resistant |
| "Compare E. coli resistance between humans and pigs in Italy" | compare |

## Evaluation Results

| Metric | Score |
|---|---|
| Intent Accuracy | 100% |
| Entity Accuracy | 100% |
| No-Hallucination Rate | 100% |
| Severity Label Accuracy | 100% |
| Answer Quality (LLM judge) | 76.7% |
| **Overall** | **95.3%** |

## Tech Stack

| Component | Technology |
|---|---|
| LLM (intent + narrative) | Ollama cloud (gemma3:4b / gemma3:27b) |
| Structured data store | SQLite + pandas |
| Vector search | ChromaDB + all-MiniLM-L6-v2 |
| Dashboard | Streamlit + Plotly |
| REST API | FastAPI + uvicorn |
| Literature retrieval | NCBI E-utilities + BioPython |

## Limitations

- EFSA animal/food chain data currently limited to 2015 (single timepoint)
- No ESBL/AmpC-specific organism data
- No serovar-level Salmonella resistance data
- Trend analysis limited for animal data (single year)
- Dependence on cloud inference introduces availability risk

## Citation

If you use AMR Sentinel in your research, please cite: <paper_link>

## Acknowledgements

- ECDC and EFSA for open AMR surveillance data
- ChromaDB for vector storage
- Streamlit for the dashboard framework
- Ollama for cloud model hosting
MDEOF

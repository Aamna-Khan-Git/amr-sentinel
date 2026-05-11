# 🦠 AMR Sentinel

> **An agentic AI assistant for European Antimicrobial Resistance (AMR) surveillance — grounded in real EARS-Net/EFSA data, not hallucinations.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Anthropic Claude](https://img.shields.io/badge/powered%20by-Claude%20API-orange.svg)](https://www.anthropic.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-purple.svg)](https://www.trychroma.com/)

---

## Why AMR Sentinel?

General-purpose LLMs **cannot** be safely used for AMR surveillance queries:

| Problem with general LLMs | How AMR Sentinel solves it |
|---|---|
| Knowledge cutoff — resistance rates change every year | Queries live SQLite DB loaded from your EARS-Net/EFSA CSV |
| Confidently hallucinate specific resistance percentages | Only cites `[DATA]` when rows actually exist in the database |
| Generic advice regardless of actual resistance levels | Recommendations triggered by real thresholds (CRITICAL ≥50%, HIGH ≥25%) |
| No access to private institutional surveillance data | All data stays local — never sent to a third party |
| Cannot search recent literature automatically | Semantic search over indexed PubMed abstracts via ChromaDB |

---

## Features

- **Natural language querying** — ask questions in plain English, get grounded answers
- **Live surveillance data** — SQLite database loaded from EARS-Net/EFSA tidy CSV
- **Semantic literature search** — PubMed abstracts embedded with `all-MiniLM-L6-v2`
- **Actionable recommendations** — clinical, public health, veterinary, and policy actions based on actual resistance levels
- **Interactive Plotly dashboard** — trend lines, country comparisons, bubble charts, KPI metrics
- **REST API** — FastAPI `/ask` endpoint for programmatic access
- **Resistance severity flagging** — 🔴 CRITICAL / 🟠 HIGH / 🟡 MODERATE / 🟢 LOW
- **ML pipeline** — clustering, feature importance, and resistance prediction (`amr_pipeline.py`)

---

## Project Structure

```
amr-sentinel/
├── agents/
│   ├── data_agent.py             <- SQL query dispatcher (trend/compare/top_resistant)
│   ├── literature_agent.py       <- ChromaDB semantic search wrapper
│   └── narrative_agent.py        <- Claude API: grounded narrative + recommendations
├── amr_outputs/
│   ├── amr_tidy.csv              <- Processed tidy surveillance data
│   ├── cluster_assignments.csv   <- ML clustering results
│   ├── feature_importances.csv   <- ML feature importance scores
│   ├── model_metrics.json        <- ML model evaluation metrics
│   ├── predictions.csv           <- Resistance predictions
│   └── plots/                    <- Generated visualisation PNGs
├── data/
│   ├── raw/                      <- Raw EFSA/EARS-Net source files
│   ├── processed/                <- Intermediate processed files
│   ├── embeddings/               <- ChromaDB vector store
│   └── abstracts.jsonl           <- PubMed abstracts (fetched via NCBI)
├── amr_pipeline.py               <- ML pipeline: clustering + prediction
├── api_server.py                 <- Streamlit dashboard + FastAPI REST endpoint
├── config.py                     <- Config loader (reads from .env)
├── db_setup.py                   <- CSV -> SQLite loader + query helpers
├── fetch_abstracts_ncbi.py       <- NCBI E-utilities abstract fetcher
├── literature_store.py           <- ChromaDB ingest + semantic search
├── orchestrator.py               <- Intent parser + agent router
├── requirements.txt
├── .env.example                  <- Template for secrets
└── README.md
```

---

## Architecture

```
User question
      |
      v
orchestrator.ask()
      |
      |-> Claude Haiku  ->  parse intent + extract params as JSON
      |                     (organism, antibiotic, country, year)
      |
      |-> data_agent    ->  SQL query on SQLite
      |                     (trend / compare / top_resistant)
      |
      |-> lit_agent     ->  ChromaDB cosine search
      |                     (top-k PubMed abstracts)
      |
      └-> Claude Sonnet ->  synthesise grounded narrative
                            + actionable recommendations
```

---

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
nano .env   # add your API keys
```

Required variables:

```
ANTHROPIC_API_KEY=sk-ant-...
NCBI_EMAIL=your@email.com
```

### 3. Load surveillance data

Place your tidy AMR CSV at `data/amr_tidy.csv` with columns:
`country`, `year`, `matrix`, `organism`, `antibiotic`, `percent_resistant`, `n_isolates`

```bash
python3 db_setup.py
```

### 4. Fetch and index PubMed abstracts

```bash
python3 fetch_abstracts_ncbi.py        # fetches from NCBI into data/abstracts.jsonl
python3 literature_store.py data/abstracts.jsonl   # embeds into ChromaDB
```

### 5. Launch the dashboard

```bash
streamlit run api_server.py -- --ui
```

Or the REST API:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

---

## Usage

### Streamlit Dashboard

Open the Network URL printed by Streamlit (e.g. `http://10.x.x.x:8501`).

Example questions to try:

| Question | Intent triggered |
|---|---|
| "How has ciprofloxacin resistance in E. coli changed over time?" | `trend` |
| "Compare tetracycline resistance in Salmonella across countries" | `compare` |
| "Which organism/antibiotic combinations have the highest resistance?" | `top_resistant` |
| "How has ampicillin resistance in E. faecalis trended?" | `trend` |
| "Compare GEN resistance in C. jejuni across Europe" | `compare` |

### REST API

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Compare ciprofloxacin resistance in E. coli across Europe", "k_literature": 5}'
```

**Response schema:**
```json
{
  "question": "...",
  "intent": "compare",
  "answer": "## Summary\n...\n### Recommended Actions\n...",
  "data": { "rows": [...], "row_count": 27, "error": null },
  "literature": { "hits": [...], "hit_count": 5, "store_size": 199 },
  "narrative_metadata": { "model": "claude-sonnet-4-5", "input_tokens": 900, "output_tokens": 650 }
}
```

---

## Supported Data

### Organisms
`C. coli`, `C. jejuni`, `E. coli`, `E. coli (ESBL/AmpC)`, `E. faecalis`, `E. faecium`,
`S. Derby`, `S. Enteritidis`, `S. Infantis`, `S. Kentucky`, `S. Typhimurium`,
`S. Typhimurium (mono)`, `Salmonella spp.`

### Antibiotic Codes
`AMC`, `AMK`, `AMP`, `AZM`, `AmpC_PHENO`, `CAZ`, `CHL`, `CIP`, `COL`, `CTX`,
`DPT`, `ERY`, `ESBL_AmpC_PHENO`, `ESBL_PHENO`, `ETP`, `GEN`, `LZD`, `MEM`,
`NAL`, `QDA`, `SMX`, `SXT`, `TEC`, `TET`, `TGC`, `TMP`, `VAN`

---

## ML Pipeline

`amr_pipeline.py` runs a standalone machine learning analysis:

- **Clustering** — groups country/organism/antibiotic profiles by resistance patterns
- **Feature importance** — identifies which factors most predict resistance
- **Resistance prediction** — trains a model to predict `percent_resistant`

Outputs saved to `amr_outputs/`:
```bash
python3 amr_pipeline.py
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `NCBI_API_KEY` | No | — | NCBI key (higher rate limits) |
| `NCBI_EMAIL` | No | — | Email for NCBI E-utilities |
| `MODEL_NAME` | No | `claude-sonnet-4-5` | Claude model identifier |
| `DATABASE_PATH` | No | `amr_sentinel.db` | SQLite database path |
| `CHROMA_PATH` | No | `chroma_store` | ChromaDB persistent store path |

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM (intent parsing) | Claude Haiku (`claude-haiku-4-5`) |
| LLM (narrative + recommendations) | Claude Sonnet (`claude-sonnet-4-5`) |
| Structured data store | SQLite + pandas |
| Vector search | ChromaDB + `all-MiniLM-L6-v2` |
| ML pipeline | scikit-learn (clustering, regression) |
| Dashboard | Streamlit + Plotly |
| REST API | FastAPI + uvicorn |
| Literature retrieval | NCBI E-utilities + BioPython |

---

## Limitations

- Surveillance data quality and coverage vary by country and year
- Literature search quality depends on the size and recency of the indexed abstract collection
- Resistance thresholds (CRITICAL/HIGH/MODERATE/LOW) are indicative — clinical decisions should always involve a qualified microbiologist
- No patient-level data is processed; all data is aggregated surveillance statistics
- The system does not replace official ECDC/EFSA surveillance reports

---

## Data Sources

- [ECDC EARS-Net](https://www.ecdc.europa.eu/en/antimicrobial-resistance/surveillance-and-disease-data/data-ecdc) — European Antimicrobial Resistance Surveillance Network
- [EFSA](https://www.efsa.europa.eu/en/data-report/antimicrobial-resistance-zoonotic-bacteria-animals-food-and-humans) — European Food Safety Authority AMR data
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/) — biomedical literature via NCBI E-utilities

---

## Acknowledgements

- [Anthropic](https://www.anthropic.com/) for the Claude API
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the dashboard framework
- [ECDC](https://www.ecdc.europa.eu/) and [EFSA](https://www.efsa.europa.eu/) for open AMR surveillance data

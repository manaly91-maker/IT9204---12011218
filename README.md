# Gen Z Marketing Decision Assistant
IT9204 Emerging Technologies in AI — Bahrain Polytechnic, Semester B 2025-2026
Student: Manal Talal Ameen | ID: 12011218

## System Overview
Cloud-based AI system on Microsoft Azure combining:
- RAG pipeline (Azure AI Search — genz-insight-index, 6 chunks)
- AutoML classification (Azure ML — AUC 0.99994)
- gpt-4o agent with 3 tools
- Deterministic decision router (Paths A/B/C/D)

## Files
- config.py — Azure credentials (replace with your own keys)
- rag_pipeline.py — indexes documents into Azure AI Search
- agent.py — gpt-4o agent with tool calling
- decision_logic.py — keyword-based query router
- main.py — system entry point

## Datasets
- Survey dataset: https://www.kaggle.com/datasets/vrindhamoka/generation-z-and-decision-making
- Spending dataset: https://www.kaggle.com/datasets/manandkumar/gen-z-money-spending-dataset

## Demo Video
[INSERT LINK HERE]

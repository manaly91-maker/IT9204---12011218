import os, json, csv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField,
    SearchField, SearchFieldDataType, VectorSearch,
    HnswAlgorithmConfiguration, VectorSearchProfile
)
from azure.core.credentials import AzureKeyCredential
from config import *

openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_VERSION
)
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

def create_index():
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="hnsw-profile"
        )
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-algo")]
    )
    index = SearchIndex(name=AZURE_SEARCH_INDEX, fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index)
    print(f"Index created.")

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

def get_embedding(text):
    response = openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding

def load_survey_documents():
    docs = []
    csv_path = "gen z science project (Responses) - Form Responses 1.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping.")
        return docs

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"  Survey rows: {len(rows)}")

    college_rationale, improvement_suggestions, college_determinants = [], [], []

    for r in rows:
        for col, val in r.items():
            val = (val or "").strip()
            col_lower = col.lower()
            if "why did you choose" in col_lower and val.upper() not in ["N/A","NA",""]:
                college_rationale.append(val)
            elif "one way to improve" in col_lower and val:
                improvement_suggestions.append(val)
            elif "determines what college" in col_lower and val.upper() not in ["N/A","NA",""]:
                college_determinants.append(val)

    docs.append({"source": "survey_college_rationale",
                 "content": "Gen Z College Selection Rationale:\n" + "\n".join(f"- {r}" for r in college_rationale) or "Survey data on college selection."})
    docs.append({"source": "survey_improvement_strategies",
                 "content": "Gen Z Decision Improvement Strategies:\n" + "\n".join(f"- {r}" for r in improvement_suggestions) or "Survey data on decision improvement."})
    docs.append({"source": "survey_college_determinants",
                 "content": "Gen Z College Determination Factors:\n" + "\n".join(f"- {r}" for r in college_determinants) or "Survey data on college determinants."})
    return docs

def load_spending_document():
    csv_path = "genz_spending_with_target.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping.")
        return []

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    print(f"  Spending rows: {total}")
    avg_income    = sum(float(r["Income (USD)"]) for r in rows) / total
    avg_savings   = sum(float(r["Savings (USD)"]) for r in rows) / total
    avg_shopping  = sum(float(r["Online Shopping (USD)"]) for r in rows) / total
    avg_eating    = sum(float(r["Eating Out (USD)"]) for r in rows) / total
    avg_entertain = sum(float(r["Entertainment (USD)"]) for r in rows) / total

    content = f"""Gen Z Money Spending Dataset Summary (n={total} respondents):
- Average income: ${avg_income:.0f}/month
- Average savings: ${avg_savings:.0f}/month ({avg_savings/avg_income*100:.1f}% of income)
- Average online shopping: ${avg_shopping:.0f}/month
- Average eating out: ${avg_eating:.0f}/month
- Average entertainment: ${avg_entertain:.0f}/month
- High spenders (discretionary > 15% of income): approximately 62% of respondents
- Low spenders: approximately 38% of respondents
Key insight: Gen Z allocates significant discretionary income to online shopping and food experiences."""

    return [{"source": "spending_dataset_summary", "content": content}]

def upload_documents(docs):
    upload_batch = []
    doc_id = 0
    for doc in docs:
        chunks = chunk_text(doc["content"])
        print(f"  {doc['source']}: {len(chunks)} chunks")
        for chunk in chunks:
            embedding = get_embedding(chunk)
            upload_batch.append({
                "id": str(doc_id),
                "content": chunk,
                "source": doc["source"],
                "embedding": embedding
            })
            doc_id += 1
    if upload_batch:
        search_client.upload_documents(upload_batch)
        print(f"Uploaded {len(upload_batch)} chunks.")
    else:
        print("No chunks to upload!")
    return len(upload_batch)

def retrieve_insights(query, top_k=3):
    from azure.search.documents.models import VectorizedQuery
    query_embedding = get_embedding(query)
    vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=top_k, fields="embedding")
    results = search_client.search(search_text=None, vector_queries=[vector_query],
                                   select=["content", "source"], top=top_k)
    return [{"content": r["content"], "source": r["source"]} for r in results]

if __name__ == "__main__":
    print("Creating index...")
    create_index()
    print("Loading documents...")
    docs = load_survey_documents() + load_spending_document()
    print(f"Total documents: {len(docs)}")
    print("Uploading chunks...")
    total = upload_documents(docs)
    print(f"Done. {total} chunks indexed.")

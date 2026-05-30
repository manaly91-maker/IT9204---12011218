import json
from datetime import datetime

# ── Keyword Signal Sets ────────────────────────────────────────────
CAMPAIGN_SIGNALS      = ["campaign", "strategy", "market", "advertise", "platform", "instagram", "tiktok", "messaging", "target", "audience"]
CLASSIFICATION_SIGNALS= ["predict", "classify", "profile", "spender", "high or low", "spending type", "income", "features", "how much"]
INSIGHT_SIGNALS       = ["what do", "why do", "how do", "explain", "describe", "survey", "respondent", "behaviour", "pattern", "gen z say"]

def classify_query(query):
    """Classify a query into routing path A, B, C, or D."""
    q = query.lower()

    campaign_score      = sum(1 for s in CAMPAIGN_SIGNALS       if s in q)
    classification_score= sum(1 for s in CLASSIFICATION_SIGNALS if s in q)
    insight_score       = sum(1 for s in INSIGHT_SIGNALS        if s in q)

    if campaign_score >= 2:
        path = "C"
        reason = f"Campaign signals detected: {campaign_score}"
    elif classification_score >= 2 and classification_score > insight_score:
        path = "A"
        reason = f"Classification signals detected: {classification_score}"
    elif insight_score >= 1 and insight_score >= classification_score:
        path = "B"
        reason = f"Insight signals detected: {insight_score}"
    else:
        path = "D"
        reason = "Ambiguous query — defaulting to RAG fallback"

    log_entry = {
        "timestamp":  datetime.now().isoformat(),
        "query":      query,
        "path":       path,
        "reason":     reason,
        "scores":     {"campaign": campaign_score, "classification": classification_score, "insight": insight_score}
    }

    with open("routing_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return path, reason

if __name__ == "__main__":
    test_queries = [
        "What do Gen Z respondents say about choosing their college?",
        "Predict spending profile: age 22, income 5000, shopping 400, eating 300, entertainment 200, savings 800",
        "Generate an Instagram campaign strategy for high-spending Gen Z students",
        "Something seems off about this segment — can you help?",
    ]
    for q in test_queries:
        path, reason = classify_query(q)
        print(f"Query: {q[:60]}...")
        print(f"  → Path {path}: {reason}\n")

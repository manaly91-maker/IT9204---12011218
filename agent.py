import json, requests
from openai import AzureOpenAI
from config import *
from rag_pipeline import retrieve_insights

# ── Client ─────────────────────────────────────────────────────────
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_VERSION
)

# ── Tool Definitions ───────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_insights",
            "description": "Retrieve relevant insights from Gen Z survey and spending documents using vector search. Use this for qualitative behavioural questions about Gen Z decision-making, college choices, or spending patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search the Gen Z knowledge base"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_spending_profile",
            "description": "Predict whether a Gen Z respondent is a High or Low spender based on their financial features. Calls the real Azure ML AutoML model (AUC 0.99994). Returns prediction label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "age":            {"type": "number", "description": "Age of respondent"},
                    "income":         {"type": "number", "description": "Monthly income in USD"},
                    "online_shopping":{"type": "number", "description": "Monthly online shopping spend in USD"},
                    "eating_out":     {"type": "number", "description": "Monthly eating out spend in USD"},
                    "entertainment":  {"type": "number", "description": "Monthly entertainment spend in USD"},
                    "savings":        {"type": "number", "description": "Monthly savings in USD"}
                },
                "required": ["age", "income", "online_shopping", "eating_out", "entertainment", "savings"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_campaign_strategy",
            "description": "Generate a structured digital marketing campaign strategy for a Gen Z segment on a specific platform.",
            "parameters": {
                "type": "object",
                "properties": {
                    "profile":  {"type": "string", "description": "Description of the Gen Z segment (e.g. High Spender, female, age 22, college student)"},
                    "platform": {"type": "string", "description": "Target platform: Instagram, TikTok, Snapchat, or LinkedIn"}
                },
                "required": ["profile", "platform"]
            }
        }
    }
]

# ── System Prompt ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the Gen Z Marketing Decision Assistant — an AI system helping marketing managers understand Gen Z consumer behaviour and create targeted campaigns.

You have access to three tools:
1. retrieve_insights — searches Gen Z survey and spending documents. Use this for any qualitative question about Gen Z behaviour, college decisions, or spending patterns.
2. predict_spending_profile — calls an Azure ML model to classify a Gen Z respondent as High or Low Spender. Use when the user provides financial feature values.
3. generate_campaign_strategy — generates a structured marketing campaign for a given Gen Z segment and platform.

ROUTING RULES:
- Qualitative insight questions ("What do Gen Z say about...", "Why do they...") → retrieve_insights
- Classification questions with features provided → predict_spending_profile; if confidence ≥ 0.70, also call retrieve_insights
- Campaign/strategy questions → retrieve_insights first, then generate_campaign_strategy
- Complex questions combining insight + prediction + strategy → call all relevant tools in sequence

IMPORTANT:
- Never fabricate Gen Z behaviour claims. All assertions must be grounded in retrieved documents.
- Always cite the source document when using retrieved evidence.
- Always disclose the ML prediction when making spending classifications.
- You are a decision-support tool. The marketing manager makes the final decision."""

# ── Tool Execution ─────────────────────────────────────────────────
def execute_tool(tool_name, tool_args, audit_log):
    """Execute a tool call and return the result as a string."""

    if tool_name == "retrieve_insights":
        results = retrieve_insights(tool_args["query"])
        audit_log.append({"tool": "retrieve_insights", "query": tool_args["query"], "chunks_returned": len(results)})
        if not results:
            return "No relevant documents found."
        output = []
        for r in results:
            output.append(f"[Source: {r['source']}]\n{r['content']}")
        return "\n\n---\n\n".join(output)

    elif tool_name == "predict_spending_profile":
        try:
            # ── Build input payload matching AutoML training columns ──
            input_data = {
                "input_data": {
                    "columns": [
                        "ID",
                        "Age",
                        "Income (USD)",
                        "Rent (USD)",
                        "Groceries (USD)",
                        "Eating Out (USD)",
                        "Entertainment (USD)",
                        "Subscription Services (USD)",
                        "Education (USD)",
                        "Online Shopping (USD)",
                        "Savings (USD)",
                        "Investments (USD)",
                        "Travel (USD)",
                        "Fitness (USD)",
                        "Miscellaneous (USD)"
                    ],
                    "data": [[
                        1,
                        tool_args["age"],
                        tool_args["income"],
                        0,
                        0,
                        tool_args["eating_out"],
                        tool_args["entertainment"],
                        0,
                        0,
                        tool_args["online_shopping"],
                        tool_args["savings"],
                        0,
                        0,
                        0,
                        0
                    ]]
                }
            }

            # ── Call the real Azure ML endpoint ──
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AZURE_ML_KEY}"
            }

            response = requests.post(
                AZURE_ML_ENDPOINT,
                headers=headers,
                json=input_data,
                timeout=30
            )
            response.raise_for_status()
            prediction = response.json()

            # ── Parse response (AutoML returns a list e.g. [1] or [0]) ──
            if isinstance(prediction, list):
                predicted_class = int(prediction[0])
            elif isinstance(prediction, dict) and "result" in prediction:
                predicted_class = int(prediction["result"][0])
            else:
                predicted_class = int(prediction)

            label = "High Spender" if predicted_class == 1 else "Low Spender"

            result = {
                "predicted_class": predicted_class,
                "label": label,
                "model": "Azure AutoML — TruncatedSVDWrapper + LogisticRegression",
                "model_auc": 0.99994,
                "segment": f"{'High' if predicted_class == 1 else 'Low'} Spender",
                "marketing_implication": (
                    "Use aspirational, premium messaging with exclusive offers and FOMO-driven CTAs."
                    if predicted_class == 1 else
                    "Use value-focused messaging emphasising savings, quality, and long-term benefits."
                )
            }
            audit_log.append({"tool": "predict_spending_profile", "result": result})
            return json.dumps(result)

        except requests.exceptions.ConnectionError:
            return json.dumps({"error": "Cannot reach Azure ML endpoint. Please ensure genz-spending-endpoint is deployed and AZURE_ML_ENDPOINT/AZURE_ML_KEY are set in config.py."})
        except requests.exceptions.HTTPError as e:
            return json.dumps({"error": f"Azure ML endpoint returned HTTP error: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"ML prediction error: {str(e)}"})

    elif tool_name == "generate_campaign_strategy":
        profile  = tool_args["profile"]
        platform = tool_args["platform"]
        strategies = {
            "Instagram": {"format": "Carousel posts and Reels", "tone": "Visual, aspirational, authentic"},
            "TikTok":    {"format": "Short-form video (15-30s)", "tone": "Entertaining, trending, relatable"},
            "Snapchat":  {"format": "Stories and Snap Ads",      "tone": "Fun, ephemeral, peer-driven"},
            "LinkedIn":  {"format": "Thought leadership posts",  "tone": "Professional, career-focused, insightful"},
        }
        s = strategies.get(platform, {"format": "Mixed content", "tone": "Engaging and authentic"})
        strategy = {
            "campaign_objective": "Increase Gen Z brand engagement and conversion",
            "target_segment": profile,
            "platform": platform,
            "messaging_tone": s["tone"],
            "content_format": s["format"],
            "content_examples": [
                "'Real students, real choices' — authentic testimonial series",
                "Behind-the-scenes brand content showing Gen Z values",
                "Interactive polls and challenges tied to decision-making themes"
            ],
            "hashtags": ["#GenZ", "#RealTalk", "#SmartChoices", f"#{platform}Marketing"],
            "cta_text": "Discover what Gen Z is choosing — join the conversation",
            "budget_tier": "Mid-tier ($500-2000/month for student subscription)"
        }
        audit_log.append({"tool": "generate_campaign_strategy", "profile": profile, "platform": platform})
        return json.dumps(strategy, indent=2)

    return f"Unknown tool: {tool_name}"

# ── Agent Loop ─────────────────────────────────────────────────────
def answer_query(user_query, conversation_history=None):
    """Run the agentic loop for a user query."""
    if conversation_history is None:
        conversation_history = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += conversation_history
    messages.append({"role": "user", "content": user_query})

    audit_log = []
    max_iterations = 5

    for iteration in range(max_iterations):
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                result    = execute_tool(tool_name, tool_args, audit_log)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
        else:
            final_response = msg.content
            return {"response": final_response, "audit_log": audit_log}

    return {"response": "Max iterations reached.", "audit_log": audit_log}

# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Gen Z Marketing Decision Assistant")
    print("Azure ML Model: TruncatedSVDWrapper + LogisticRegression")
    print("AUC: 0.99994 | Endpoint: genz-spending-endpoint")
    print("=" * 60)
    print("Type 'quit' to exit.\n")

    history = []
while True:
    query = input("You: ").strip()
    if query.lower() == "quit":
        break
        result = answer_query(query, history)
        print(f"\nAssistant: {result['response']}")
        print(f"[Tools used: {[l['tool'] for l in result['audit_log']]}]\n")
        history.append({"role": "user",      "content": query})
        history.append({"role": "assistant", "content": result["response"]})

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification, pipeline
import os
import sqlite3

app = FastAPI(
    title="BizDecisions AI Backend",
    description="FastAPI service for customer review aspect-based sentiment analysis.",
    version="1.0.0"
)

# -------------------------------------------------------------------
# MODEL LOADING (Lazy initialization on startup)
# -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None
model = None
zeroshot_classifier = None

@app.on_event("startup")
async def startup_event():
    global tokenizer, model, zeroshot_classifier
    try:
        model_path = os.getenv("MODEL_PATH", r"C:\new\my_bert_model")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
        model.to(device)
        model.eval()
        
        # Load default zeroshot pipeline
        zeroshot_classifier = pipeline(
            "zero-shot-classification", 
            model="cross-encoder/nli-distilroberta-base", 
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning loading models: {e}")

# -------------------------------------------------------------------
# SCHEMAS
# -------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    review: str
    aspects: Optional[List[str]] = None

class BatchAnalyzeRequest(BaseModel):
    reviews: List[str]
    aspects: Optional[List[List[str]]] = None

class AspectResult(BaseModel):
    aspect: str
    sentiment: str
    confidence: float
    action: str

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------
ADVICE_DB = {
    "product": {"Negative": "Check product quality/inventory.", "Positive": "Promote products."},
    "service": {"Negative": "Conduct staff training.", "Positive": "Reward staff."},
    "price": {"Negative": "Review pricing strategy.", "Positive": "Pricing is effective."},
    "ambience": {"Negative": "Improve layout/cleanliness.", "Positive": "Environment appreciated."}
}

def analyze_single_aspect(text: str, aspect: str) -> AspectResult:
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    mapped_aspect = "product" if aspect.lower() == "food" else aspect.lower()
    
    inputs = tokenizer(
        text,
        text_pair=mapped_aspect,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    idx = int(torch.argmax(probs))
    confidence = float(probs[0][idx])

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = label_map[idx]
    
    if confidence < 0.5:
        sentiment = "Uncertain"
        
    action = ADVICE_DB.get(mapped_aspect, {}).get(sentiment, "No action needed")
    
    return AspectResult(
        aspect=aspect,
        sentiment=sentiment,
        confidence=confidence,
        action=action
    )

# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.get("/health")
async def health_check():
    status = "healthy" if model and tokenizer else "models_loading"
    return {"status": status, "device": str(device)}

@app.post("/analyze", response_model=List[AspectResult])
async def analyze_review(req: AnalyzeRequest):
    """Analyze a single review for specific aspects, or auto-detect if none provided."""
    target_aspects = req.aspects
    if not target_aspects:
        if not zeroshot_classifier:
            target_aspects = ["product"]
        else:
            labels = ["product", "service", "price", "ambience"]
            res = zeroshot_classifier(req.review, candidate_labels=labels, multi_label=True)
            target_aspects = [l for l, s in zip(res['labels'], res['scores']) if s > 0.4]
            if not target_aspects:
                target_aspects = [res['labels'][0]]
                
    results = []
    for asp in target_aspects:
        results.append(analyze_single_aspect(req.review, asp))
    return results

@app.post("/analyze-batch")
async def analyze_batch(req: BatchAnalyzeRequest):
    """Analyze multiple reviews (simplistic sequential implementation for brevity)."""
    # In a full prod app this would batch via torch inputs directly
    responses = []
    for i, review in enumerate(req.reviews):
        aspects = req.aspects[i] if req.aspects and i < len(req.aspects) else None
        
        # Build individual request
        single_req = AnalyzeRequest(review=review, aspects=aspects)
        try:
            res = await analyze_review(single_req)
            responses.append({"review": review, "results": res})
        except Exception as e:
            responses.append({"review": review, "error": str(e)})
            
    return responses

@app.get("/history")
async def get_history(limit: int = 100):
    """Fetch the latest saved results from the centralized DB."""
    try:
        conn = sqlite3.connect("business_intelligence.db", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM reviews_history ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in c.fetchall()]
        return {"count": len(rows), "data": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

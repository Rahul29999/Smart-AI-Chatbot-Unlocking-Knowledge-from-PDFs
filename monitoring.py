
import json
import os
from datetime import datetime
from typing import Optional

LOG_FILE = "query_logs.json"

def log_query(
    query: str,
    answer: str,
    source: str,
    latency_ms: Optional[float] = None
) -> None:
    log = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "source": source,
        "latency_ms": latency_ms
    }
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    logs.append(log)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

def get_stats() -> dict:
    if not os.path.exists(LOG_FILE):
        return {"total_queries": 0}
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)
    total = len(logs)
    pdf_count = sum(1 for l in logs if l["source"] == "pdf")
    gpt_count = sum(1 for l in logs if l["source"] == "gpt")
    avg_latency = sum(
        l["latency_ms"] for l in logs if l["latency_ms"]
    ) / max(total, 1)
    return {
        "total_queries": total,
        "pdf_answered": pdf_count,
        "gpt_answered": gpt_count,
        "avg_latency_ms": round(avg_latency, 2)
    }

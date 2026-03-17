from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

app = FastAPI(title="Yik Hak'd API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Path to your model output CSV ───────────────────────────────────────────
CSV_PATH = os.getenv("CSV_PATH", "predicted_output1.csv")

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}
PROFANITY_MAP = {"🟢": "green", "🟡": "yellow", "🔴": "red"}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])
    return df


def format_post(row) -> dict:
    return {
        "post_text":       row["text"],
        "likes":           int(row["VOTES"]),
        "timestamp":       (row["DATE"] - pd.Timedelta(hours=5)).isoformat(),
        "sentiment":       SENTIMENT_MAP.get(int(row["prediction"]), "neutral"),
        "profanity_color": PROFANITY_MAP.get(str(row["profanity_emoji"]), "green"),
    }


@app.get("/top-posts")
def top_posts(limit: int = 5):
    df = load_data()
    cutoff = df["DATE"].max() - pd.Timedelta(days=30)
    df = df[df["DATE"] >= cutoff]
    top = df.sort_values(["VOTES", "DATE"], ascending=[False, False]).head(limit)
    return [format_post(row) for _, row in top.iterrows()]


@app.get("/active-times")
def active_times():
    df = load_data()
    df["hour"] = (df["DATE"] - pd.Timedelta(hours=5)).dt.hour
    counts = df.groupby("hour").size().reset_index(name="count")
    all_hours = pd.DataFrame({"hour": range(24)})
    counts = all_hours.merge(counts, on="hour", how="left").fillna(0)
    counts["count"] = counts["count"].astype(int)
    counts["label"] = counts["hour"].apply(
        lambda h: f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"
    )
    return counts.to_dict(orient="records")


@app.get("/search")
def search(keyword: str = Query(..., min_length=1)):
    df = load_data()
    mask = df["text"].str.contains(keyword, case=False, na=False)
    results = df[mask]
    posts = [format_post(row) for _, row in results.iterrows()]
    return {
        "positive": [p for p in posts if p["sentiment"] == "positive"],
        "neutral":  [p for p in posts if p["sentiment"] == "neutral"],
        "negative": [p for p in posts if p["sentiment"] == "negative"],
        "total":    len(posts),
    }


@app.get("/stats")
def stats():
    df = load_data()
    return {
        "total_posts":  len(df),
        "last_updated": df["DATE"].max().isoformat(),
        "avg_votes":    round(df["VOTES"].mean(), 1),
    }


@app.get("/trending-topic")
def trending_topic():
    import re
    from collections import Counter

    STOPWORDS = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "is","it","its","this","that","was","are","be","been","have","has","had",
        "do","did","does","not","no","so","if","as","by","from","up","out","about",
        "i","you","he","she","we","they","my","your","his","her","our","their",
        "me","him","us","them","who","what","when","where","how","all","just",
        "like","get","got","can","will","would","could","should","really","very",
        "one","two","more","some","any","there","here","now","then","than","been",
        "also","even","still","back","into","over","after","before","because",
        "too","much","going","know","think","want","need","make","see","go","come",
        "actually","literally","honestly","idk","lol","omg","wtf","ngl","fr","rn",
        "dont","cant","wont","im","u","ur","bc","tbh","imo","gonna","gotta",
        "wanna","yeah","yea","nah","ok","okay","hey","hi","wow","people","thing",
        "things","time","year","day","week","always","never","every","something",
        "everyone","anyone","someone","anything","everything","nothing","thats",
        "theyre","whats","campus","mercyhurst","class","classes","students","upvote",
        "just","said","good","great","bad","well","real","sure","true","feel","felt",
        "went","used","took","made","were","which","these","those","with","that",
        "have","your","what","from","they","when","this","their","could","would",
    }

    df = load_data()
    cutoff = df["DATE"].max() - pd.Timedelta(days=30)
    df = df[df["DATE"] >= cutoff]

    words = []
    for text in df["text"]:
        tokens = re.findall(r"\b[a-zA-Z]{4,}\b", str(text).lower())
        words.extend([w for w in tokens if w not in STOPWORDS])

    if not words:
        return {"topic": None}

    most_common = Counter(words).most_common(1)[0][0]
    return {"topic": most_common.capitalize()}


@app.get("/debug")
def debug():
    return {
        "csv_path":     CSV_PATH,
        "csv_exists":   os.path.exists(CSV_PATH),
        "cwd":          os.getcwd(),
        "files_in_cwd": os.listdir("."),
    }


@app.get("/debug-posts")
def debug_posts():
    try:
        df = load_data()
        return {"rows": len(df), "columns": df.columns.tolist(), "sample": df.head(2).to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}


# ── Serve React frontend ──────────────────────────────────────────────────────
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BUILD_DIR = os.path.join(os.path.dirname(__file__), "../frontend/dist")

if os.path.exists(BUILD_DIR):
    if os.path.exists(BUILD_DIR + "/assets"):
        app.mount("/assets", StaticFiles(directory=BUILD_DIR + "/assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        return FileResponse(BUILD_DIR + "/index.html")
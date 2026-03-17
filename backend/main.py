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

# ── Column mappings from your CSV → what the frontend expects ───────────────
#   text            → post_text
#   VOTES           → likes
#   DATE            → timestamp
#   prediction      → sentiment  (0=negative, 1=neutral, 2=positive)
#   profanity_emoji → profanity_color  (🟢=green, 🟡=yellow, 🔴=red)

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
    """Convert a CSV row into the shape the React frontend expects."""
    return {
        "post_text":      row["text"],
        "likes":          int(row["VOTES"]),
        "timestamp":      row["DATE"].isoformat(),
        "sentiment":      SENTIMENT_MAP.get(int(row["prediction"]), "neutral"),
        "profanity_color": PROFANITY_MAP.get(str(row["profanity_emoji"]), "green"),
    }


# ── /top-posts ───────────────────────────────────────────────────────────────
@app.get("/top-posts")
def top_posts(limit: int = 5):
    """Top N posts by votes from the last 30 days, most recent first on ties."""
    df = load_data()
    cutoff = df["DATE"].max() - pd.Timedelta(days=5)
    df = df[df["DATE"] >= cutoff]
    top = (
        df.sort_values(["VOTES", "DATE"], ascending=[False, False])
        .head(limit)
    )
    return [format_post(row) for _, row in top.iterrows()]


# ── /active-times ─────────────────────────────────────────────────────────────
@app.get("/active-times")
def active_times():
    """Post counts grouped by hour-of-day for the activity bar chart."""
    df = load_data()
    df["hour"] = (df["DATE"] - pd.Timedelta(hours=5)).dt.hour
    counts = df.groupby("hour").size().reset_index(name="count")

    # fill any missing hours with 0
    all_hours = pd.DataFrame({"hour": range(24)})
    counts = all_hours.merge(counts, on="hour", how="left").fillna(0)
    counts["count"] = counts["count"].astype(int)
    counts["label"] = counts["hour"].apply(
        lambda h: f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"
    )
    return counts.to_dict(orient="records")


# ── /search ──────────────────────────────────────────────────────────────────
@app.get("/search")
def search(keyword: str = Query(..., min_length=1)):
    """Search posts by keyword, grouped by sentiment."""
    df = load_data()
    mask = df["text"].str.contains(keyword, case=False, na=False)
    results = df[mask].head(limit)

    posts = [format_post(row) for _, row in results.iterrows()]

    return {
        "positive": [p for p in posts if p["sentiment"] == "positive"],
        "neutral":  [p for p in posts if p["sentiment"] == "neutral"],
        "negative": [p for p in posts if p["sentiment"] == "negative"],
        "total":    len(posts),
    }


# ── /stats ────────────────────────────────────────────────────────────────────
@app.get("/stats")
def stats():
    """Quick summary stats."""
    df = load_data()
    return {
        "total_posts":  len(df),
        "last_updated": df["DATE"].max().isoformat(),
        "avg_votes":    round(df["VOTES"].mean(), 1),
    }

@app.get("/debug")
def debug():
    import os
    return {
        "csv_path": CSV_PATH,
        "csv_exists": os.path.exists(CSV_PATH),
        "cwd": os.getcwd(),
        "files_in_cwd": os.listdir("."),
    }

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

BUILD_DIR = os.path.join(os.path.dirname(__file__), "../frontend/dist")

if os.path.exists(BUILD_DIR):
    app.mount("/assets", StaticFiles(directory=BUILD_DIR + "/assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        return FileResponse(BUILD_DIR + "/index.html")
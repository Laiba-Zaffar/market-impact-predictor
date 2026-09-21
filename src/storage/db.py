import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "news_sentiment.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    source TEXT,
    time_published TEXT NOT NULL,
    overall_sentiment_score REAL,
    overall_sentiment_label TEXT,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS ticker_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    ticker TEXT NOT NULL,
    relevance_score REAL,
    ticker_sentiment_score REAL,
    ticker_sentiment_label TEXT,
    UNIQUE(article_id, ticker)
);

CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_key TEXT NOT NULL,
    month_key TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    item_count INTEGER NOT NULL,
    UNIQUE(batch_key, month_key)
);

CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS article_topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    topic TEXT NOT NULL,
    relevance_score REAL NOT NULL,
    UNIQUE(article_id, topic)
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def insert_article(conn: sqlite3.Connection, article: dict) -> int | None:
    try:
        cur = conn.execute(
            """INSERT INTO articles (url, title, source, time_published, overall_sentiment_score,
                                      overall_sentiment_label, summary)
               VALUES (:url, :title, :source, :time_published, :overall_sentiment_score,
                       :overall_sentiment_label, :summary)""",
            article,
        )
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def insert_ticker_sentiment(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO ticker_sentiment
               (article_id, ticker, relevance_score, ticker_sentiment_score, ticker_sentiment_label)
           VALUES (:article_id, :ticker, :relevance_score, :ticker_sentiment_score, :ticker_sentiment_label)""",
        row,
    )


def is_window_fetched(conn: sqlite3.Connection, batch_key: str, month_key: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM fetch_log WHERE batch_key = ? AND month_key = ?", (batch_key, month_key)
    ).fetchone()
    return row is not None


def mark_window_fetched(conn: sqlite3.Connection, batch_key: str, month_key: str, item_count: int, fetched_at: str) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO fetch_log (batch_key, month_key, fetched_at, item_count)
           VALUES (?, ?, ?, ?)""",
        (batch_key, month_key, fetched_at, item_count),
    )


def insert_price(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO prices (ticker, date, open, high, low, close, volume)
           VALUES (:ticker, :date, :open, :high, :low, :close, :volume)""",
        row,
    )


def get_prices(conn: sqlite3.Connection, ticker: str) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date ASC", (ticker,)
    ).fetchall()


def insert_article_topic(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO article_topics (article_id, topic, relevance_score)
           VALUES (:article_id, :topic, :relevance_score)""",
        row,
    )


def get_all_article_topics(conn: sqlite3.Connection) -> dict[int, dict[str, float]]:
    """All (article_id -> {topic: relevance_score}) in one query, not one
    per article - same N+1 mistake as the price-lookup bug, avoided this
    time by fetching everything up front."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT article_id, topic, relevance_score FROM article_topics").fetchall()
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        result.setdefault(row["article_id"], {})[row["topic"]] = row["relevance_score"]
    return result

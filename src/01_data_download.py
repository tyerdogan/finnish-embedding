import os
import csv
import gzip
import io
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Gutenberg publishes the full catalog as a single gzip'd CSV — no pagination needed
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
# Always write to <project_root>/data/ regardless of where the script is invoked from
DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/raw")
WORKERS     = 10
BAR_WIDTH   = 50


# ── display ───────────────────────────────────────────────────────────────────

def _bar(ratio):
    """Unicode block progress bar of fixed width."""
    filled = int(BAR_WIDTH * max(0.0, min(1.0, ratio)))
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def _eta(elapsed, done, total):
    if done <= 0:
        return "--:--"
    secs = int(elapsed / done * (total - done))
    return f"{secs // 60}m{secs % 60:02d}s"


def _speed(elapsed, done):
    if elapsed <= 0 or done <= 0:
        return "--/s"
    r = done / elapsed
    return f"{r:.1f}/s" if r >= 1 else f"{1/r:.0f}s/book"


class Display:
    """
    Thread-safe in-place terminal display.
    Uses ANSI escape codes to overwrite the same lines on each update.
    """

    def __init__(self):
        self._lines = 0
        self._lock  = threading.Lock()

    def render(self, lines):
        with self._lock:
            if self._lines:
                print(f"\033[{self._lines}A", end="")
            for line in lines:
                print(f"\033[2K{line}")
            self._lines = len(lines)

    def clear(self):
        with self._lock:
            if self._lines:
                print(f"\033[{self._lines}A\033[J", end="")
            self._lines = 0


# ── network ───────────────────────────────────────────────────────────────────

def get_with_retry(url, retries=8, backoff=4.0):
    """
    GET with exponential back-off retry on timeouts and 5xx errors.
    Raises immediately on 4xx (permanent errors like 404).
    """
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError):
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)
        except requests.exceptions.HTTPError as exc:
            # 4xx errors are permanent — no point retrying
            if exc.response.status_code < 500:
                raise
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)


# ── catalog ───────────────────────────────────────────────────────────────────

def fetch_finnish_books():
    """
    Downloads Gutenberg's full catalog CSV (one ~5 MB gzip file) and
    returns all Finnish plain-text entries. This replaces paginating
    through a third-party API and is far more reliable.
    """
    disp = Display()
    disp.render([
        "  Downloading Gutenberg catalog (one file, ~5 MB)...",
        f"  {_bar(0)}  0 KB",
    ])

    r = requests.get(CATALOG_URL, timeout=120, stream=True)
    r.raise_for_status()

    # Stream download so we can show byte-level progress
    chunks     = []
    received   = 0
    total_size = int(r.headers.get("Content-Length", 0))

    for chunk in r.iter_content(chunk_size=65536):
        chunks.append(chunk)
        received += len(chunk)
        ratio = received / total_size if total_size else 0
        disp.render([
            "  Downloading Gutenberg catalog...",
            f"  {_bar(ratio)}  {received // 1024} / {total_size // 1024} KB",
        ])

    disp.clear()

    # Decompress and parse the CSV entirely in memory
    raw    = b"".join(chunks)
    buf    = io.BytesIO(raw)
    with gzip.open(buf, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        books  = [
            {
                "id":      int(row["Text#"]),
                "title":   row.get("Title", "Unknown"),
                "authors": row.get("Authors", ""),
            }
            for row in reader
            if row.get("Language", "").strip() == "fi"
            and row.get("Type", "").strip() == "Text"
        ]

    return books


# ── download ──────────────────────────────────────────────────────────────────

def download_book(book):
    """
    Downloads the UTF-8 plain text for a single Gutenberg book.
    Uses the canonical ebooks/{id}.txt.utf-8 URL.
    Returns 'ok', 'exists', or 'no_txt'.
    """
    book_id = book["id"]
    # Sanitize title: remove path separators and newlines that would corrupt the filename
    title   = book["title"].replace("/", "-").replace("\n", " ").replace("\r", " ").strip()
    url     = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{book_id}_{title[:60]}.txt")

    if os.path.exists(path):
        return "exists"

    try:
        r = get_with_retry(url)
    except requests.exceptions.HTTPError as exc:
        # 404 means this book has no plain-text edition — not a real error
        if exc.response.status_code == 404:
            return "no_txt"
        raise

    # Write raw bytes to preserve UTF-8 encoding (Finnish ä, ö, å)
    with open(path, "wb") as f:
        f.write(r.content)

    return "ok"


def download_all_books(books):
    """
    Downloads all books concurrently with WORKERS threads.
    Shows a single live progress bar that updates in place.
    """
    total  = len(books)
    saved  = skipped = errors = done = 0
    lock   = threading.Lock()
    start  = time.time()
    disp   = Display()
    errs   = []

    def show(title=""):
        elapsed = time.time() - start
        ratio   = done / total if total else 0
        short   = (title[:46] + "…") if len(title) > 46 else title
        disp.render([
            f"  {_bar(ratio)}  {ratio * 100:.0f}%",
            f"  {done}/{total}   saved:{saved}  skip:{skipped}  errors:{errors}   {_speed(elapsed, done)}   ETA:{_eta(elapsed, done, total)}",
            f"  → {short}",
        ])

    show("starting...")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(download_book, b): b for b in books}
        for future in as_completed(futures):
            book  = futures[future]
            title = book.get("title", "?")
            try:
                status = future.result()
                with lock:
                    done += 1
                    if status in ("ok", "exists"):
                        saved += 1
                    else:
                        skipped += 1
                    show(title)
            except Exception as exc:
                with lock:
                    done += 1
                    errors += 1
                    errs.append(f"{title[:60]} — {exc}")
                    show(title)

    elapsed = int(time.time() - start)
    disp.clear()
    print(f"\n  Done in {elapsed // 60}m{elapsed % 60:02d}s")
    print(f"  saved:{saved}   skipped (no txt):{skipped}   errors:{errors}\n")
    if errs:
        print("  Errors:")
        for e in errs[:10]:
            print(f"    {e}")
        print()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    """
    Downloads all Finnish plain-text books from Project Gutenberg.
    Reads the full catalog in one request instead of paginating an API.
    """
    books = fetch_finnish_books()

    if not books:
        print("No Finnish books found. Check your internet connection.")
        return

    print(f"  Found {len(books)} Finnish books.\n")
    download_all_books(books)


if __name__ == "__main__":
    main()

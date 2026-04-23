"""
scrape_standings.py — Robust IPL 2026 Points Table Scraper
==========================================================
HTML (Cricbuzz) → fallback → API

Usage:
    python scrape_standings.py
    python scrape_standings.py --dry-run
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

STANDINGS_PATH = Path("standings.json")

SERIES_URL = "https://www.cricbuzz.com/cricket-series/9241/indian-premier-league-2026/points-table"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cricbuzz.com/",
}

TEAM_ALIASES = {
    "mi": "Mumbai Indians",
    "csk": "Chennai Super Kings",
    "rcb": "Royal Challengers Bengaluru",
    "dc": "Delhi Capitals",
    "kkr": "Kolkata Knight Riders",
    "pbks": "Punjab Kings",
    "rr": "Rajasthan Royals",
    "srh": "Sunrisers Hyderabad",
    "gt": "Gujarat Titans",
    "lsg": "Lucknow Super Giants",
}


def norm(name):
    if not name:
        return name
    return TEAM_ALIASES.get(name.strip().lower(), name.strip())


# ─────────────────────────────────────────────────────────────
# FETCH HTML
# ─────────────────────────────────────────────────────────────
def fetch_page(url):
    try:
        log.info(f"Fetching HTML: {url}")
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        time.sleep(1.2)
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        log.warning(f"HTML fetch failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# PARSE HTML (NEW GRID LAYOUT)
# ─────────────────────────────────────────────────────────────
def parse_html_standings(soup):
    rows = []

    grid_rows = soup.select("div.point-table-grid.p-2")

    if not grid_rows:
        log.warning("No grid rows found.")
        return []

    log.info(f"Found {len(grid_rows)} rows in HTML")

    for row in grid_rows:
        cols = row.find_all("div", recursive=False)

        try:
            # Team is inside nested <a><div class="text-xs">PBKS</div>
            team_tag = row.select_one("a div.text-xs")
            if not team_tag:
                continue

            team = norm(team_tag.get_text(strip=True))

            nums = [c.get_text(strip=True) for c in cols]

            rows.append({
                "team": team,
                "M": int(nums[2]),
                "W": int(nums[3]),
                "L": int(nums[4]),
                "NR": int(nums[5]),
                "Pts": int(nums[6]),
                "NRR": float(nums[7].replace("+", "")),
                "Form": "",
            })

        except Exception as e:
            log.debug(f"Skipping row: {e}")
            continue

    log.info(f"Parsed {len(rows)} teams from HTML")
    return rows

# ─────────────────────────────────────────────────────────────
# API FALLBACK
# ─────────────────────────────────────────────────────────────
def fetch_api_standings():
    url = "https://www.cricbuzz.com/api/cricket-series/9241/points-table"

    try:
        log.info("Trying API fallback...")
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()

        rows = []

        for group in data.get("pointsTable", []):
            for t in group.get("pointsTableInfo", []):
                rows.append({
                    "team": norm(t.get("teamName")),
                    "M": t.get("matchesPlayed"),
                    "W": t.get("matchesWon"),
                    "L": t.get("matchesLost"),
                    "NR": t.get("noResult"),
                    "Pts": t.get("points"),
                    "NRR": round(float(t.get("netRunRate", 0)), 3),
                    "Form": "",
                })

        log.info(f"Parsed {len(rows)} teams from API")
        return rows

    except Exception as e:
        log.warning(f"API failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# MAIN SCRAPER
# ─────────────────────────────────────────────────────────────
def scrape_standings():
    # 1. Try HTML
    soup = fetch_page(SERIES_URL)
    if soup:
        rows = parse_html_standings(soup)
        if rows:
            log.info("Using HTML data ✔")
            return rows

    # 2. Fallback → API
    rows = fetch_api_standings()
    if rows:
        log.info("Using API data ✔")
        return rows

    # 3. Fail
    log.error("All methods failed ❌")
    return []


# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
def save_standings(rows):
    data = {
        "scraped_at": __import__("datetime").datetime.utcnow().isoformat(),
        "standings": rows,
    }
    STANDINGS_PATH.write_text(json.dumps(data, indent=2))
    log.info(f"Saved → {STANDINGS_PATH}")


# ─────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────
def display(rows):
    print(f"\n{'#':>3} {'Team':<30} {'M':>3} {'W':>3} {'L':>3} {'NR':>3} {'Pts':>4} {'NRR':>7}")
    print("-" * 70)
    for i, r in enumerate(rows, 1):
        print(f"{i:>3}. {r['team']:<30} {r['M']:>3} {r['W']:>3} {r['L']:>3} {r['NR']:>3} {r['Pts']:>4} {r['NRR']:>+7.3f}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = scrape_standings()

    if not rows:
        log.error("No standings scraped.")
        return

    display(rows)

    if not args.dry_run:
        save_standings(rows)
    else:
        log.info("[DRY RUN] Not saving.")


if __name__ == "__main__":
    main()
"""
cleanup_duplicates.py
=====================
Removes duplicate 2026 match entries caused by the scraper inserting
matches that were already in the DB from the Kaggle dataset.

Run:
    python cleanup_duplicates.py --db ipl.db --dry-run   # preview
    python cleanup_duplicates.py --db ipl.db              # actual fix
"""

import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path("ipl.db")


def show_2026_state(conn):
    rows = conn.execute("""
        SELECT match_id, date, team1, team2, winner, source
        FROM matches WHERE season = '2026'
        ORDER BY date ASC, match_id ASC
    """).fetchall()
    print(f"\nTotal 2026 rows in DB: {len(rows)}")
    print(f"{'match_id':<30} {'date':<12} {'team1':<30} {'team2':<30} {'winner':<30} source")
    print("-" * 140)
    for r in rows:
        print(f"{str(r[0]):<30} {str(r[1]):<12} {str(r[2]):<30} {str(r[3]):<30} {str(r[4]):<30} {r[5]}")


def find_duplicates(conn):
    """
    Find match pairs where the same two teams played on the same date
    but have different match_ids (one from Kaggle, one from scraper).
    """
    rows = conn.execute("""
        SELECT match_id, date, team1, team2, winner, source
        FROM matches WHERE season = '2026'
        ORDER BY date ASC
    """).fetchall()

    # Group by (date, frozenset of teams)
    seen = {}
    duplicates_to_delete = []

    for row in rows:
        match_id, date, team1, team2, winner, source = row
        key = (date, frozenset([team1, team2]))

        if key in seen:
            # We have a duplicate — keep the scraped_2026/cricdata one,
            # delete the older Kaggle one (it has less accurate 2026 data)
            existing_id, existing_source = seen[key]

            # Prefer cricdata_2026 > scraped_2026 > kaggle
            priority = {"cricdata_2026": 0, "scraped_2026": 1, "kaggle": 2}
            existing_priority = priority.get(existing_source, 99)
            current_priority  = priority.get(source, 99)

            if current_priority <= existing_priority:
                # Current row is better — delete the existing one
                duplicates_to_delete.append(existing_id)
                seen[key] = (match_id, source)
            else:
                # Existing row is better — delete current one
                duplicates_to_delete.append(match_id)
        else:
            seen[key] = (match_id, source)

    return duplicates_to_delete


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate 2026 matches from ipl.db")
    parser.add_argument("--db",      default=str(DB_PATH))
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    print("=== BEFORE CLEANUP ===")
    show_2026_state(conn)

    duplicates = find_duplicates(conn)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Found {len(duplicates)} duplicate rows to delete:")
    for mid in duplicates:
        row = conn.execute(
            "SELECT match_id, date, team1, team2, source FROM matches WHERE match_id=?", (mid,)
        ).fetchone()
        if row:
            print(f"  DELETE: {row[0]:<30} {row[1]}  {row[2]} vs {row[3]}  [{row[4]}]")

    if not args.dry_run and duplicates:
        for mid in duplicates:
            conn.execute("DELETE FROM matches WHERE match_id=?", (mid,))
        conn.commit()
        print(f"\nDeleted {len(duplicates)} duplicate rows.")

        print("\n=== AFTER CLEANUP ===")
        show_2026_state(conn)

        total = conn.execute("SELECT COUNT(*) FROM matches WHERE season='2026'").fetchone()[0]
        print(f"\nFinal 2026 match count: {total}")
    elif args.dry_run:
        print("\nDry run — nothing deleted. Run without --dry-run to apply.")
    else:
        print("\nNo duplicates found — DB is clean.")

    conn.close()


if __name__ == "__main__":
    main()
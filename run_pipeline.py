import subprocess
import sys
import time

DB_PATH = "ipl.db"


def run(cmd):
    print(f"\n🚀 Running: {cmd}")
    start = time.time()

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"❌ Failed: {cmd}")
        sys.exit(1)

    print(f"✅ Done in {time.time() - start:.2f}s")


def main():
    print("\n================ IPL PIPELINE (LOCAL) ================\n")

    # 1. Scrape matches
    run(f"python scraper/scrapper_data.py --db {DB_PATH}")

    # 2. Scrape standings
    run("python scrape_standings.py")

    # 3. Feature engineering
    run(f"python features/features.py --db {DB_PATH}")

    # 4. Train model
    run(f"python model/train.py --db {DB_PATH}")

    # 5. Run simulation
    run(f"python -m simulate.simulate --db {DB_PATH} --n 10000")

    # 6. Log results
    run(f"python logger.py --db {DB_PATH}")

    print("\n🎉 PIPELINE COMPLETE\n")


if __name__ == "__main__":
    main()
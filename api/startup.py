import logging
import requests
import zipfile
import time
from pathlib import Path
from api.core.config import DB_PATH, MODELS_DIR, ARTIFACTS_URL

log = logging.getLogger(__name__)


# def ensure_artifacts():
#     """
#     Ensure ipl.db + models/ are available.

#     - If present locally → skip
#     - If missing → download from GitHub Release
#     """

#     db_ok = DB_PATH.exists()
#     models_ok = MODELS_DIR.exists() and any(MODELS_DIR.iterdir())

#     if db_ok and models_ok:
#         log.info("Artifacts already present — skipping download.")
#         return

#     if not ARTIFACTS_URL:
#         raise RuntimeError(
#             "Artifacts missing and ARTIFACTS_URL not set. "
#             "Run pipeline locally or configure backend properly."
#         )

#     _download_with_retry(ARTIFACTS_URL)

def ensure_artifacts():
    """
    Always refresh artifacts on startup.
    """

    if not ARTIFACTS_URL:
        raise RuntimeError("ARTIFACTS_URL not set")

    log.info("Refreshing artifacts from GitHub Release...")
    _download_with_retry(ARTIFACTS_URL)


def _download_with_retry(url: str, max_retries: int = 3):
    """
    Download and extract artifacts.zip with retry logic.
    Designed for cold starts (Render / cloud).
    """

    zip_path = Path("artifacts.zip")

    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"Downloading artifacts (attempt {attempt}/{max_retries})...")
            log.info(f"URL: {url}")

            response = requests.get(url, timeout=180, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            size_mb = downloaded / (1024 * 1024)
            log.info(f"Downloaded {size_mb:.2f} MB")

            log.info("Extracting artifacts...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(".")

            zip_path.unlink(missing_ok=True)

            log.info("Artifacts ready.")
            return

        except Exception as e:
            log.warning(f"Attempt {attempt} failed: {e}")

            if zip_path.exists():
                zip_path.unlink()

            if attempt < max_retries:
                wait = attempt * 10
                log.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to download artifacts after {max_retries} attempts. "
                    f"Last error: {e}"
                )
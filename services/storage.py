import os
import shutil
import aiofiles
from pathlib import Path
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "forensics")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize Supabase client
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


async def save_upload(
    job_id: str,
    filename: str,
    content: bytes,
    content_type: str | None = None,
) -> dict:
    """
    Saves file to local disk (for analysis) AND uploads to Supabase (for persistence).
    Returns a dict with 'local_path' and 'public_url'.
    """
    # 1. Local Save
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    local_path = job_dir / filename
    
    async with aiofiles.open(local_path, "wb") as f:
        await f.write(content)

    public_url = None

    # 2. Supabase Upload
    if supabase:
        try:
            storage_path = f"{job_id}/{filename}"
            supabase.storage.from_(SUPABASE_BUCKET).upload(
                path=storage_path,
                file=content,
                file_options={"content-type": content_type or "application/octet-stream"}
            )
            # Construct public URL
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{storage_path}"
        except Exception as e:
            print(f"Supabase Upload Failed: {e}")

    return {
        "local_path": local_path,
        "public_url": public_url
    }


def cleanup_upload(job_id: str) -> None:
    """Remove the local upload directory for a completed job."""
    job_dir = UPLOAD_DIR / job_id
    shutil.rmtree(job_dir, ignore_errors=True)

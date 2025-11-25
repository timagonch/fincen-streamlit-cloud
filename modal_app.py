# modal_app.py
"""
Modal entrypoint for the FinCEN fraud project.

It gives you:
  - run_full_pipeline(): crawler -> text extractor -> summary generator
 
  - serve_streamlit(): run the Streamlit dashboard on Modal

Usage examples (from your repo root: fincen-streamlit-cloud):

  modal run modal_app.py::run_crawler
  modal run modal_app.py::run_full_pipeline
  modal serve modal_app.py::serve_streamlit
"""

import os
import subprocess
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------
# Modal image: install deps + add your project directory
# ---------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # core dependencies from your pyproject
        "bs4",
        "google-genai",
        "pillow",
        "pymupdf",
        "python-dotenv",
        "requests",
        "sentence-transformers",
        "streamlit",
        "supabase",
    )
    # include the whole repo in the container at /root/fincen-streamlit-cloud
    .add_local_dir(str(REPO_ROOT), remote_path="/root/fincen-streamlit-cloud")
)

app = modal.App("fincen-fraud-platform")


def _chdir_project_root():
    """Set working directory inside the container."""
    os.chdir("/root/fincen-streamlit-cloud")


# ---------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("fincen-secrets")],
    timeout=60 * 60,
)
def run_crawler():
    """Run the FinCEN publications crawler inside Modal."""
    _chdir_project_root()
    subprocess.run(
        ["python", "fincen_publications_crawler.py"],
        check=True,
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("fincen-secrets")],
    timeout=60 * 60,
)
def run_text_extractor():
    """Run your text extractor that fills fincen_fulltext."""
    _chdir_project_root()
    subprocess.run(
        ["python", "fincen_text_extractor.py"],  # adjust if your filename differs
        check=True,
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("fincen-secrets")],
    timeout=60 * 60,
)
def run_summary_generator():
    """Run the LLM summary generator over new docs."""
    _chdir_project_root()
    subprocess.run(
        ["python", "fincen_summary_generator.py"],
        check=True,
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("fincen-secrets")],
    timeout=3 * 60 * 60,  # full pipeline can take a while
)
def run_full_pipeline():
    """
    Orchestrate the full pipeline in one Modal job:
      crawler -> text extractor -> summary generator.
    """
    print("=== [1/3] Running crawler ===")
    run_crawler.call()

    print("=== [2/3] Extracting full text ===")
    run_text_extractor.call()

    print("=== [3/3] Generating summaries ===")
    run_summary_generator.call()

    print("Pipeline complete.")



# ---------------------------------------------------------------------
# Streamlit app on Modal
# ---------------------------------------------------------------------


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("fincen-secrets")],
)
@modal.web_server(8000)
def serve_streamlit():
    """
    Serve the Streamlit dashboard on Modal as a web endpoint.

    Access it via:
      - modal serve modal_app.py    (for iterative dev)
      - modal deploy modal_app.py   (for a long-lived URL)
    """
    _chdir_project_root()

    # Start Streamlit in the background; Modal will proxy HTTP → port 8000
    cmd = [
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.port",
        "8000",
        "--server.address",
        "0.0.0.0",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
    ]

    # Don't block this function; Modal just needs the server listening on 8000.
    subprocess.Popen(cmd)



# ---------------------------------------------------------------------
# Local entrypoint (optional helper)
# ---------------------------------------------------------------------


@app.local_entrypoint()
def main(run: str = "pipeline"):
    """
    Convenience for local testing via Modal:

      modal run modal_app.py::main -- --run pipeline
      modal run modal_app.py::main -- --run crawler
      modal run modal_app.py::main -- --run streamlit
    """
    if run == "crawler":
        run_crawler.call()
    elif run == "text":
        run_text_extractor.call()
    elif run == "summaries":
        run_summary_generator.call()
    elif run == "pipeline":
        run_full_pipeline.call()
    elif run == "streamlit":
        serve_streamlit.call()
    else:
        print(
            "Unknown mode. Use one of: crawler, text, summaries, pipeline, streamlit"
        )

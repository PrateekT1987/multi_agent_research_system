# syntax=docker/dockerfile:1

# Mars runs CrewAI, WeasyPrint and Streamlit in one image.  The build is
# intentionally single-stage: the dependency set is large and a multi-stage
# split saves little while making the runtime libraries harder to reason about.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MARS_CACHE_DIR=/data/mars/cache \
    MARS_STATE_DIR=/data/mars/runs \
    MARS_OUTPUT_DIR=/data/mars/research_output \
    MARS_DEAD_LETTER_DIR=/data/mars/dead_letters \
    MARS_OBSERVABILITY_DIR=/data/mars/traces \
    MARS_APPROVAL_DIR=/data/mars/approvals

# Runtime libraries for WeasyPrint plus curl for container health checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libcairo2 \
        libffi8 \
        libgdk-pixbuf-2.0-0 \
        libharfbuzz0b \
        libjpeg62-turbo \
        libopenjp2-7 \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

RUN useradd --create-home --uid 10001 mars \
    && mkdir -p /data/mars \
    && chown -R mars:mars /app /data/mars

USER mars

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

# The UI is the default surface.  Override the command to run `mars serve` for
# a probes-only container, or `mars research ...` for a batch job.
CMD ["streamlit", "run", "src/mars/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

# Reproducible, version-pinned image for using scenic-rs -- the artifact that
# Docker/Singularity-based pipelines (Nextflow, Snakemake, ...) consume.
#
# scenic-rs is a prebuilt abi3 wheel, so this just downloads it (and numpy +
# pandas); no Rust toolchain or compilation is needed in the image.
#
# Built and pushed to ghcr.io/<owner>/scenic-rs:<version> by the release
# workflow. To build locally:
#   docker build --build-arg VERSION=0.1.1 -t scenic-rs:0.1.1 .
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/nglaszik/SCENIC-rs"
LABEL org.opencontainers.image.description="Rust implementation of the SCENIC single-cell GRN pipeline"
LABEL org.opencontainers.image.licenses="GPL-3.0-or-later"

# Pin at build time. The release workflow passes the tag's version; an empty
# VERSION (local builds) installs the latest release from PyPI.
ARG VERSION=
RUN pip install --no-cache-dir "scenic-rs${VERSION:+==${VERSION}}"

# scenic-rs is a library; default to an interactive Python with it importable.
CMD ["python"]

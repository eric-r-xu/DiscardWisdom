# Dockerized RiichiDiscardWisdom

## Build
```bash
docker build -t riichi-discard-wisdom .
```

## Run (interactive shell with tools installed)
```bash
docker run --rm -it riichi-discard-wisdom
```

## Run the Efficiency CLI
Tenhou groups example (14 tiles):
```bash
docker run --rm -it riichi-discard-wisdom efficiency --groups m123456789p11s11
```

Tile labels example (14 tiles):
```bash
docker run --rm -it riichi-discard-wisdom efficiency --labels 1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 1p 1s 1s 1s
```

With additional visible tiles affecting remaining:
```bash
docker run --rm -it riichi-discard-wisdom efficiency --labels 1m 1m 1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 1p  --visible p111
```

JSON output:
```bash
docker run --rm riichi-discard-wisdom efficiency --groups m123456789p11s11 --json
```

## Run the Defense CLI
```bash
docker run --rm -it riichi-discard-wisdom defense --groups m123p456s789z12345
```

### docker-compose (optional)
```bash
docker compose up --build
# then inside the container:
# efficiency -h
# defense -h
```

> Note: `opencv-python` and `pytesseract` require system libraries that are installed in the image. If you need languages beyond English for OCR, install the relevant tesseract language packs by extending the Dockerfile (e.g., `apt-get install tesseract-ocr-jpn`).

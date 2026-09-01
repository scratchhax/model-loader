#!/usr/bin/env bash
# One-shot: adds the model-loader service to your existing docker-compose.yaml
# and brings it up. Idempotent — safe to re-run.
#
# Usage: from the directory that holds your docker-compose.yaml, run:
#   bash /path/to/model_loader/bootstrap.sh

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="$(pwd)"
COMPOSE_FILE=""

for candidate in docker-compose.yaml docker-compose.yml compose.yaml compose.yml; do
  if [ -f "$COMPOSE_DIR/$candidate" ]; then
    COMPOSE_FILE="$COMPOSE_DIR/$candidate"
    break
  fi
done

if [ -z "$COMPOSE_FILE" ]; then
  echo "No docker-compose file found in $COMPOSE_DIR"
  echo "Run this from your compose project directory (the one with docker-compose.yaml)."
  exit 1
fi

echo "==> Using compose file: $COMPOSE_FILE"

# Symlink or ensure a local ./model_loader dir points at this repo
if [ ! -e "$COMPOSE_DIR/model_loader" ]; then
  echo "==> Linking ./model_loader -> $REPO_DIR"
  ln -s "$REPO_DIR" "$COMPOSE_DIR/model_loader"
fi

if grep -q '^\s*model-loader:' "$COMPOSE_FILE"; then
  echo "==> model-loader service already present in $COMPOSE_FILE"
else
  echo "==> Adding model-loader service block"
  # Insert the snippet before any top-level `volumes:` line, or append at end
  python3 - <<PYEOF
from pathlib import Path
p = Path("$COMPOSE_FILE")
src = p.read_text()
snippet = Path("$REPO_DIR/docker-compose.snippet.yaml").read_text()
# strip leading comment block from snippet
lines_body = [l for l in snippet.splitlines(keepends=True) if not l.lstrip().startswith("#")]
snippet_body = "".join(lines_body).lstrip("\n")

if "\nvolumes:" in src or src.startswith("volumes:"):
    # insert before top-level "volumes:"
    lines = src.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.startswith("volumes:"):
            lines.insert(i, "\n" + snippet_body + "\n")
            break
    p.write_text("".join(lines))
else:
    p.write_text(src.rstrip() + "\n\n" + snippet_body + "\n")
print("  inserted.")
PYEOF
fi

mkdir -p "$COMPOSE_DIR/model_loader_data"

echo "==> Building + starting model-loader"
docker compose -f "$COMPOSE_FILE" up -d --build model-loader

echo
echo "Done. App should be at http://$(hostname -I | awk '{print $1}'):8090/"
echo "If this is a fresh host with existing GGUFs in /models, run the layout migration once:"
echo "  docker exec model-loader python3 -m app.migrate_layout"

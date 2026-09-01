#!/usr/bin/env bash
# Snapshot the important small state into a single tarball.
# Does NOT include the GGUFs themselves — those are big, back them up separately.
#
# Usage: from the compose project directory, run:
#   bash /path/to/model_loader/backup.sh [dest_dir]

set -e

COMPOSE_DIR="$(pwd)"
DEST="${1:-$COMPOSE_DIR/backups}"
mkdir -p "$DEST"
TS="$(date +%Y%m%d-%H%M%S)"
OUT="$DEST/model-loader-state-$TS.tgz"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> Collecting state into $TMP"

# 1) app sqlite prefs (HF token, prompts, avatar cache, download history)
if [ -f "$COMPOSE_DIR/model_loader_data/model_loader.db" ]; then
  cp "$COMPOSE_DIR/model_loader_data/model_loader.db" "$TMP/model_loader.db"
  echo "   + model_loader.db"
fi

# 2) models.ini + its rolling backups
if [ -f "$COMPOSE_DIR/models/models.ini" ]; then
  mkdir -p "$TMP/models_ini"
  cp "$COMPOSE_DIR/models/models.ini" "$TMP/models_ini/"
  cp "$COMPOSE_DIR/models"/models.ini.bak-* "$TMP/models_ini/" 2>/dev/null || true
  echo "   + models.ini and $(ls "$TMP/models_ini"/models.ini.bak-* 2>/dev/null | wc -l) backups"
fi

# 3) OpenWebUI sqlite (if present) — has your chats, user settings, connection prefixes
if [ -f "$COMPOSE_DIR/webui_data/webui.db" ]; then
  cp "$COMPOSE_DIR/webui_data/webui.db" "$TMP/webui.db"
  echo "   + webui.db"
fi

# 4) compose file
for candidate in docker-compose.yaml docker-compose.yml compose.yaml compose.yml; do
  if [ -f "$COMPOSE_DIR/$candidate" ]; then
    cp "$COMPOSE_DIR/$candidate" "$TMP/"
    echo "   + $candidate"
    break
  fi
done

tar -C "$TMP" -czf "$OUT" .
SIZE="$(du -h "$OUT" | cut -f1)"
echo "==> Wrote $OUT ($SIZE)"

# rotate: keep newest 20
ls -1t "$DEST"/model-loader-state-*.tgz 2>/dev/null | tail -n +21 | xargs -r rm -v

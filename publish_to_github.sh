#!/usr/bin/env bash
# One-time publish script. Run after: gh auth login
set -euo pipefail

cd "$(dirname "$0")"
GH="${GH:-$HOME/.local/bin/gh}"
REPO_NAME="face-mask-detection"

if ! command -v "$GH" >/dev/null 2>&1; then
  echo "Install GitHub CLI first: https://cli.github.com/"
  exit 1
fi

"$GH" auth status || { echo "Run: $GH auth login"; exit 1; }

USER=$("$GH" api user -q .login)
echo "Publishing as $USER..."

# Create repo if it does not exist
if ! "$GH" repo view "$USER/$REPO_NAME" >/dev/null 2>&1; then
  "$GH" repo create "$REPO_NAME" \
    --public \
    --description "Face mask detection with OpenCV SSD + MobileNetV2" \
    --source . \
    --remote origin \
    --push
else
  git remote remove origin 2>/dev/null || true
  git remote add origin "https://github.com/$USER/$REPO_NAME.git"
  git push -u origin main
fi

"$GH" repo edit "$USER/$REPO_NAME" \
  --add-topic face-mask-detection \
  --add-topic opencv \
  --add-topic tensorflow \
  --add-topic computer-vision \
  --add-topic deep-learning

echo ""
echo "Done: https://github.com/$USER/$REPO_NAME"

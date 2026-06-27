#!/bin/zsh
export PATH="/opt/homebrew/bin:$HOME/praisonai-audio-editor/.venv/bin:$PATH"
[ -z "$OPENAI_API_KEY" ] && source "$HOME/.zshrc" 2>/dev/null
[ -z "$OPENAI_API_KEY" ] && source "$HOME/.zprofile" 2>/dev/null

for f in "$@"; do
  praisonai-editor edit "$f" --preset podcast --verbose 2>&1 | tee "${f%.*}_edit_log.txt"
done

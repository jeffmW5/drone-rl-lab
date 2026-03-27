#!/usr/bin/env bash
export PATH="/home/jeff/.nvm/versions/node/v20.20.1/bin:$PATH"
exec /usr/bin/gnome-terminal -- /bin/bash -lc 'export PATH="/home/jeff/.nvm/versions/node/v20.20.1/bin:$PATH"; /home/jeff/.nvm/versions/node/v20.20.1/bin/codex --dangerously-bypass-approvals-and-sandbox; exec bash'

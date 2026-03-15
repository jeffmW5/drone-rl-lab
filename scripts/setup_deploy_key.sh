#!/bin/bash
# =============================================================================
# Drone RL Lab — GitHub Deploy Key Setup
# =============================================================================
# Creates an SSH key scoped ONLY to the drone-rl-lab repo.
# Safer than using your personal GitHub token on a cloud machine.
#
# Usage: bash scripts/setup_deploy_key.sh
# =============================================================================

KEY_PATH="$HOME/.ssh/drone_rl_lab_deploy"

if [ -f "$KEY_PATH" ]; then
    echo "Deploy key already exists at $KEY_PATH"
    echo "Public key:"
    cat "${KEY_PATH}.pub"
    exit 0
fi

# Generate key
mkdir -p ~/.ssh
ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "runpod-drone-rl-lab"

# Configure SSH to use this key for GitHub
cat >> ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/drone_rl_lab_deploy
    IdentitiesOnly yes
EOF

echo ""
echo "================================================"
echo "  DEPLOY KEY CREATED"
echo ""
echo "  Step 1: Copy the public key below"
echo "  Step 2: Go to: https://github.com/jeffmW5/drone-rl-lab/settings/keys"
echo "  Step 3: Click 'Add deploy key'"
echo "  Step 4: Paste the key, check 'Allow write access', save"
echo ""
echo "  PUBLIC KEY:"
echo "  ─────────────────────────────────────────────"
cat "${KEY_PATH}.pub"
echo "  ─────────────────────────────────────────────"
echo ""
echo "  Then test with: ssh -T git@github.com"
echo "================================================"
echo ""

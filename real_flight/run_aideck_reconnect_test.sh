#!/usr/bin/env bash
set -u

cd /home/jeff/drone-rl-lab || exit 1

echo "AI Deck Reconnect Test"
echo "======================"
echo
echo "This test repeatedly opens TCP, reads exactly one frame, closes TCP,"
echo "waits one second, and tries again."
echo

python3.11 real_flight/aideck_reconnect_test.py
status=$?

echo
echo "Test finished with exit code ${status}."
echo "Logs are under: /home/jeff/drone-rl-lab/real_flight/aideck_logs/"
echo
read -r -p "Press Enter to close this window..."
exit "$status"

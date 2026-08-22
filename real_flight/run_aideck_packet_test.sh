#!/usr/bin/env bash
set -u

cd /home/jeff/drone-rl-lab || exit 1

echo "AI Deck Packet Test"
echo "==================="
echo
echo "This test expects the VM/network to be connected to the AI Deck WiFi AP."
echo "Default target: 192.168.4.1:5000, fallback: 192.168.7.201:5000"
echo

python3.11 real_flight/aideck_packet_test.py
status=$?

echo
echo "Test finished with exit code ${status}."
echo "Logs are under: /home/jeff/drone-rl-lab/real_flight/aideck_logs/"
echo
read -r -p "Press Enter to close this window..."
exit "$status"

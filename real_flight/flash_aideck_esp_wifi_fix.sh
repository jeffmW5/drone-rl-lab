#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jeff/drone-rl-lab"
FIRMWARE_DIR="$ROOT_DIR/real_flight/firmware/esp_wifi_transport_fix_20260516"
LOG_DIR="$ROOT_DIR/real_flight/aideck_logs/flash_$(date +%Y%m%d_%H%M%S)"

OPENOCD="${AIDECK_OPENOCD:-$HOME/Downloads/openocd-esp32/bin/openocd}"
OPENOCD_SCRIPTS="${AIDECK_OPENOCD_SCRIPTS:-$HOME/Downloads/openocd-esp32/share/openocd/scripts}"
OPENOCD_INTERFACE="${AIDECK_OPENOCD_INTERFACE:-interface/ftdi/olimex-arm-usb-tiny-h.cfg}"
OPENOCD_BOARD="${AIDECK_OPENOCD_BOARD:-board/esp32-solo-1.cfg}"
OPENOCD_SPEED="${AIDECK_OPENOCD_SPEED:-1000}"

APP_BIN="$FIRMWARE_DIR/aideck_esp.bin"
BOOTLOADER_BIN="$FIRMWARE_DIR/bootloader.bin"
PARTITION_BIN="$FIRMWARE_DIR/partitions_singleapp.bin"
MODE="${1:-app}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/flash.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "AI Deck ESP32 WiFi transport firmware flash"
echo "Log: $LOG_FILE"
echo

for path in "$OPENOCD" "$APP_BIN"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required file: $path"
    exit 1
  fi
done

if [[ "$MODE" == "full" ]]; then
  for path in "$BOOTLOADER_BIN" "$PARTITION_BIN"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing required file: $path"
      exit 1
    fi
  done
elif [[ "$MODE" != "app" ]]; then
  echo "Usage: $0 [app|full]"
  exit 1
fi

echo "Firmware:"
sha256sum "$APP_BIN"
if [[ "$MODE" == "full" ]]; then
  sha256sum "$BOOTLOADER_BIN" "$PARTITION_BIN"
fi
echo

echo "Hardware setup before continuing:"
echo "1. Keep propellers off. This only flashes the AI Deck ESP32; it should not arm motors."
echo "2. Power the Crazyflie/AI Deck so the ESP32 is powered."
echo "3. Connect the Olimex ARM-USB-TINY-H to the AI Deck ESP debug header."
echo "4. Connect the Olimex USB cable to this VM/host."
echo "5. Keep your normal internet WiFi connection; this step does not use the aideck-stream AP."
echo

if command -v lsusb >/dev/null 2>&1; then
  echo "Current likely USB debug devices:"
  lsusb | grep -Ei 'olimex|ftdi|future|15ba|0403' || echo "  No Olimex/FTDI device visible yet."
  echo
fi

read -r -p "Press Enter when the hardware is connected, or Ctrl-C to cancel. "

OPENOCD_ARGS=(
  -s "$OPENOCD_SCRIPTS"
  -f "$OPENOCD_INTERFACE"
  -f "$OPENOCD_BOARD"
  -c "adapter speed $OPENOCD_SPEED"
)

if [[ "$MODE" == "full" ]]; then
  OPENOCD_ARGS+=(
    -c "program_esp $BOOTLOADER_BIN 0x1000 verify"
    -c "program_esp $PARTITION_BIN 0x8000 verify"
    -c "program_esp $APP_BIN 0x10000 verify reset exit"
  )
else
  OPENOCD_ARGS+=(
    -c "program_esp $APP_BIN 0x10000 verify reset exit"
  )
fi

echo
echo "Running OpenOCD with $OPENOCD_BOARD at adapter speed $OPENOCD_SPEED."
"$OPENOCD" "${OPENOCD_ARGS[@]}"

echo
echo "Flash complete. Power-cycle the Crazyflie/AI Deck before running the packet test."

# GAP8 AI Deck WiFi image streamer

This is the GAP8 hyperflash image restored to the physical AI Deck over
Olimex JTAG on 2026-08-29. It runs Bitcraze's `wifi-img-streamer` configured
for:

- SSID: `aideck-stream`
- image mode: `JPEG_ENCODING`
- resolution: 324x244 grayscale

Artifact:

- `target.board.devices.flash.img`
- size: 73,760 bytes
- SHA256: `6851a619f7a7ce4f7135f7abab3b31559d039be390cd694bf3f7b304b620edbf`

Build provenance:

- source: `/home/jeff/Downloads/aideck-gap8-examples-master/examples/other/wifi-img-streamer/wifi-img-streamer.c`
- source SHA256: `5dd2d76ef70b67de2b82e14ec1389bda12744f30f203eaeba2d0552f7ea90aae`
- builder: `bitcraze/aideck:latest`
- builder digest: `sha256:038197df9cb86ccf8e6649e93dd0cf23781830e136288523983768918851633e`
- target: `ai_deck` / `GAP8_V2` / FreeRTOS

The JTAG flash completed successfully. Native-Windows SSID and frame-flow
verification still requires a physical Crazyflie/AI Deck power-cycle.

Do not replace this with a clean upstream build without checking the source:
upstream defaults have previously used RAW mode and a different SSID. Do not
flash the ESP32 TXQ16 experiment; the old stall was caused by VirtualBox
networking, not the deck firmware.

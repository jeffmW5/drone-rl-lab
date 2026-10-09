# THR1 observation protocol

CPX source GAP8, destination STM32, function APP. Payload 40 bytes, little endian.

| Offset | Type | Meaning |
|---|---|---|
| 0 | char[4] | THR1 magic |
| 4 | uint8 | version 1 |
| 5 | uint8 | bit 0: synthetic-trained model; other bits reserved |
| 6 | uint8 | class index into deployed MODEL.json classes |
| 7 | uint8 | reserved zero |
| 8 | uint32 | sequence, wraps modulo 2^32 |
| 12 | uint32 | GAP8 capture request clock in microseconds; wraps |
| 16 | uint32 | measured inference duration in microseconds |
| 20 | int32 | top-two raw-logit margin, saturating at INT32_MAX |
| 24 | int32[4] | integer logits; scale in MODEL.json |

Margin is not a calibrated probability. Class names/order belong to the model
manifest and must not be assumed by index. These outputs do not measure physical
tether tension, payload 3D position, or obstacle clearance.

Receiver telemetry reports `tether.recent` only for non-synthetic packets received
within 200 ms. This is RECENT RECEIPT, not sensor-exposure age. GAP8 and STM32 clocks
are unsynchronized; transport delay still needs measurement before control use.
The receiver checks schema, best-class consistency, margin and sequence order.
A 2-second quiet period allows sequence reset following a GAP8 reboot.

The provided receiver only logs. It never calls a commander or motor API.
Actual closed-loop handling needs a defined tether task, real-data validation,
camera geometry calibration, latency bounds and an independently tested fallback.

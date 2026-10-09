/* SPDX-License-Identifier: GPL-3.0-only
 * Camera perception prototype. Never sends flight-control setpoints.
 * Integrates Bitcraze GPL-3.0 camera/CPX modules and Apache-2.0 DORY code.
 */
#include "pmsis.h"
#include "bsp/bsp.h"
#include "camera_pipeline.h"
#include "stream_config.h"
#include "cpx.h"
#include "network.h"
#include "tether_model.h"
#include <stdint.h>
#include <string.h>
#include <limits.h>

#define ARENA_BYTES (96 * 1024)
static uint8_t *input;
static uint8_t *arena;
static int32_t logits[4];
static uint32_t sequence;

/* Pixel-center nearest-neighbor mapping. Match preprocessing.py exactly. */
static void downsample(const uint8_t *image) {
  for (unsigned y = 0; y < 64; ++y) {
    const unsigned sy = ((2 * y + 1) * CAMERA_HEIGHT) / 128;
    for (unsigned x = 0; x < 64; ++x) {
      const unsigned sx = ((2 * x + 1) * CAMERA_WIDTH) / 128;
      input[y * 64 + x] = image[sy * CAMERA_WIDTH + sx];
    }
  }
}

static void put32(uint8_t *dest, uint32_t value) {
  dest[0] = value; dest[1] = value >> 8;
  dest[2] = value >> 16; dest[3] = value >> 24;
}

static void send_perception(uint32_t started, uint32_t duration) {
  unsigned best = 0, second;
  for (unsigned i = 1; i < 4; ++i) if (logits[i] > logits[best]) best = i;
  second = best == 0 ? 1 : 0;
  for (unsigned i = 0; i < 4; ++i)
    if (i != best && logits[i] > logits[second]) second = i;
  int64_t margin = (int64_t)logits[best] - logits[second];
  if (margin > INT32_MAX) margin = INT32_MAX;
  CPXPacket_t packet = {0};
  cpxInitRoute(CPX_T_GAP8, CPX_T_STM32, CPX_F_APP, &packet.route);
  memcpy(packet.data, "THR1", 4);
  packet.data[4] = 1;
  packet.data[5] = TETHER_SYNTHETIC ? 1 : 0;
  packet.data[6] = best;
  packet.data[7] = 0;
  put32(packet.data + 8, sequence++);
  put32(packet.data + 12, started); /* capture-request clock, not exposure time */
  put32(packet.data + 16, duration);
  put32(packet.data + 20, (uint32_t)margin);
  for (unsigned i = 0; i < 4; ++i) put32(packet.data + 24 + 4*i, (uint32_t)logits[i]);
  packet.dataLength = 40;
  /* Bounded transport wait: dropped reports must never stall perception. */
  (void)cpxSendPacket(&packet, 10);
}

static void application(void *unused) {
  (void)unused;
  pi_bsp_init();
  cpxInit();
  input = pi_l2_malloc(64 * 64);
  arena = pi_l2_malloc(ARENA_BYTES);
  if (!input || !arena) {
    cpxPrintToConsole(LOG_TO_CRTP, "Tether: L2 allocation failed\n");
    pmsis_exit(-1);
    return;
  }
  CameraPipelineStatus_t status = camera_pipeline_init();
  if (status != CAMERA_PIPELINE_OK) {
    cpxPrintToConsole(LOG_TO_CRTP, "Tether camera: %s\n", camera_pipeline_status_message(status));
    pmsis_exit(-2);
    return;
  }
  while (1) {
    uint32_t started = pi_time_get_us();
    CameraFrame_t frame = camera_pipeline_begin_frame();
    downsample((const uint8_t *)frame.buffer->data);
    camera_pipeline_end_frame();
    uint32_t inference_start = pi_time_get_us();
    network_run(arena, ARENA_BYTES, logits, 0, 1, input);
    send_perception(started, pi_time_get_us() - inference_start);
    vTaskDelay(1);
  }
}

int main(void) {
  return pmsis_kickoff((void *)application);
}

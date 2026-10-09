/* SPDX-License-Identifier: GPL-3.0-only
 * Observation-only receiver. Contains no commander or motor-control calls.
 */
#include "app.h"
#include "cpx.h"
#include "cpx_internal_router.h"
#include "FreeRTOS.h"
#include "task.h"
#include "log.h"
#include "tether_protocol.h"

static TetherObservation observation;
static uint32_t received, rejected;
static uint8_t recent, class_id, synthetic;
static int32_t margin;
static uint32_t inference_us, sequence;

static void onPacket(const CPXPacket_t *packet) {
  TetherObservation candidate;
  const uint32_t now = T2M(xTaskGetTickCount());
  if(packet->route.source != CPX_T_GAP8 ||
     !tether_decode(packet->data, packet->dataLength, now, &candidate)) {
    rejected++; return;
  }
  taskENTER_CRITICAL();
  /* A quiet interval permits recovery after GAP8 reboot/sequence reset. */
  if(received && (uint32_t)(now-observation.received_ms)<2000u &&
     !tether_newer(candidate.sequence, observation.sequence)) {
    rejected++;
  } else {
    observation=candidate;
    received++;
  }
  taskEXIT_CRITICAL();
}

void appMain(void) {
  cpxRegisterAppMessageHandler(onPacket);
  while(1) {
    taskENTER_CRITICAL();
    recent=received && tether_recent(&observation, T2M(xTaskGetTickCount()));
    class_id=observation.class_id; synthetic=observation.synthetic;
    margin=observation.margin; inference_us=observation.inference_us;
    sequence=observation.sequence;
    taskEXIT_CRITICAL();
    vTaskDelay(M2T(20));
  }
}

LOG_GROUP_START(tether)
LOG_ADD(LOG_UINT8, recent, &recent)
LOG_ADD(LOG_UINT8, class, &class_id)
LOG_ADD(LOG_UINT8, synthetic, &synthetic)
LOG_ADD(LOG_INT32, margin, &margin)
LOG_ADD(LOG_UINT32, infer_us, &inference_us)
LOG_ADD(LOG_UINT32, sequence, &sequence)
LOG_ADD(LOG_UINT32, received, &received)
LOG_ADD(LOG_UINT32, rejected, &rejected)
LOG_GROUP_STOP(tether)

#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>

// ===========================
// Select camera model in board_config.h
// ===========================
#include "board_config.h"

// ===========================
// Edge Impulse FOMO model
// ===========================
#include "CE103-Project_inferencing.h"
#include "edge-impulse-sdk/dsp/image/image.hpp"

// ===========================
// Enter your WiFi credentials
// ===========================
const char *ssid = "********";
const char *password = "********";

// ===========================
// FOMO inference settings
// ===========================
#define EI_CAMERA_RAW_FRAME_BUFFER_COLS   320
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS   240
#define EI_CAMERA_FRAME_BYTE_SIZE         3

static bool debug_nn = false;
static uint8_t *snapshot_buf = nullptr;

// Shared result for web display (protected by simple flag)
static volatile int    g_human_count   = 0;
static volatile float  g_best_conf     = 0.0f;
static volatile bool   g_result_ready  = false;

// ===========================
// Web server & camera stream
// ===========================
void startCameraServer();
void setupLedFlash();

// ===========================
// FOMO camera helpers
// ===========================
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr)
{
    size_t pixel_ix   = offset * 3;
    size_t pixels_left = length;
    size_t out_ptr_ix  = 0;

    while (pixels_left != 0) {
        // Swap BGR → RGB (esp32-camera quirk)
        out_ptr[out_ptr_ix] = (snapshot_buf[pixel_ix + 2] << 16)
                            + (snapshot_buf[pixel_ix + 1] << 8)
                            +  snapshot_buf[pixel_ix];
        out_ptr_ix++;
        pixel_ix += 3;
        pixels_left--;
    }
    return 0;
}

/**
 * Capture one frame from the camera, resize it to the model input size,
 * run FOMO inference, and print results over Serial.
 */
static void run_fomo_inference()
{
    // Allocate snapshot buffer (RGB888, raw capture size)
    snapshot_buf = (uint8_t *)malloc(
        EI_CAMERA_RAW_FRAME_BUFFER_COLS *
        EI_CAMERA_RAW_FRAME_BUFFER_ROWS *
        EI_CAMERA_FRAME_BYTE_SIZE);

    if (snapshot_buf == nullptr) {
        Serial.println("[FOMO] ERR: Failed to allocate snapshot buffer");
        return;
    }

    // --- Capture JPEG frame ---
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("[FOMO] Camera capture failed");
        free(snapshot_buf);
        snapshot_buf = nullptr;
        return;
    }

    // Decode JPEG → RGB888 into snapshot_buf
    bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, snapshot_buf);
    esp_camera_fb_return(fb);

    if (!converted) {
        Serial.println("[FOMO] JPEG → RGB888 conversion failed");
        free(snapshot_buf);
        snapshot_buf = nullptr;
        return;
    }

    // --- Resize to model input (96×96) if needed ---
    if (EI_CLASSIFIER_INPUT_WIDTH  != EI_CAMERA_RAW_FRAME_BUFFER_COLS ||
        EI_CLASSIFIER_INPUT_HEIGHT != EI_CAMERA_RAW_FRAME_BUFFER_ROWS)
    {
        ei::image::processing::crop_and_interpolate_rgb888(
            snapshot_buf,
            EI_CAMERA_RAW_FRAME_BUFFER_COLS,
            EI_CAMERA_RAW_FRAME_BUFFER_ROWS,
            snapshot_buf,
            EI_CLASSIFIER_INPUT_WIDTH,
            EI_CLASSIFIER_INPUT_HEIGHT);
    }

    // --- Build signal and run classifier ---
    ei::signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data     = &ei_camera_get_data;

    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, debug_nn);

    free(snapshot_buf);
    snapshot_buf = nullptr;

    if (err != EI_IMPULSE_OK) {
        Serial.printf("[FOMO] ERR: run_classifier failed (%d)\n", err);
        return;
    }

    // --- Print timing ---
    Serial.printf("[FOMO] DSP: %d ms | Classify: %d ms\n",
                  result.timing.dsp, result.timing.classification);

    // --- Parse FOMO bounding boxes ---
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    int    human_count = 0;
    float  best_conf   = 0.0f;

    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) continue;   // skip zero-confidence cells

        human_count++;
        if (bb.value > best_conf) best_conf = bb.value;

        Serial.printf("[FOMO]  %s (%.2f) x:%u y:%u w:%u h:%u\n",
                      bb.label, bb.value,
                      bb.x, bb.y, bb.width, bb.height);
    }

    if (human_count == 0) {
        Serial.println("[FOMO] No human detected.");
    } else {
        Serial.printf("[FOMO] *** %d human(s) detected! Best conf: %.2f ***\n",
                      human_count, best_conf);
    }

    // Store for status endpoint
    g_human_count  = human_count;
    g_best_conf    = best_conf;
    g_result_ready = true;

#else
    // Fallback: classification model (should not happen for FOMO)
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        Serial.printf("[FOMO]  %s: %.5f\n",
                      ei_classifier_inferencing_categories[i],
                      result.classification[i].value);
    }
#endif
}

// ===========================
// Arduino setup
// ===========================
void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    Serial.println();

    // --- Camera config ---
    camera_config_t config;
    config.ledc_channel  = LEDC_CHANNEL_0;
    config.ledc_timer    = LEDC_TIMER_0;
    config.pin_d0        = Y2_GPIO_NUM;
    config.pin_d1        = Y3_GPIO_NUM;
    config.pin_d2        = Y4_GPIO_NUM;
    config.pin_d3        = Y5_GPIO_NUM;
    config.pin_d4        = Y6_GPIO_NUM;
    config.pin_d5        = Y7_GPIO_NUM;
    config.pin_d6        = Y8_GPIO_NUM;
    config.pin_d7        = Y9_GPIO_NUM;
    config.pin_xclk      = XCLK_GPIO_NUM;
    config.pin_pclk      = PCLK_GPIO_NUM;
    config.pin_vsync     = VSYNC_GPIO_NUM;
    config.pin_href      = HREF_GPIO_NUM;
    config.pin_sccb_sda  = SIOD_GPIO_NUM;
    config.pin_sccb_scl  = SIOC_GPIO_NUM;
    config.pin_pwdn      = PWDN_GPIO_NUM;
    config.pin_reset     = RESET_GPIO_NUM;
    config.xclk_freq_hz  = 20000000;

    // JPEG for streaming; FOMO will decode on the fly
    config.pixel_format  = PIXFORMAT_JPEG;
    config.grab_mode     = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location   = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality  = 12;
    config.fb_count      = 1;

    if (psramFound()) {
        config.frame_size  = FRAMESIZE_UXGA;
        config.fb_location = CAMERA_FB_IN_PSRAM;
    } else {
        config.frame_size  = FRAMESIZE_SVGA;
        config.fb_location = CAMERA_FB_IN_DRAM;
        config.fb_count    = 1;
    }

    // Camera init
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s->id.PID == OV3660_PID) {
        s->set_vflip(s, 1);
        s->set_brightness(s, 1);
        s->set_saturation(s, -2);
    }
    // Lower resolution initially for better framerate
    s->set_framesize(s, FRAMESIZE_QVGA);

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
    s->set_vflip(s, 1);
    s->set_hmirror(s, 1);
#endif
#if defined(CAMERA_MODEL_ESP32S3_EYE)
    s->set_vflip(s, 1);
#endif

#if defined(LED_GPIO_NUM)
    setupLedFlash();
#endif

    // --- WiFi ---
    WiFi.begin(ssid, password);
    WiFi.setSleep(false);

    Serial.print("WiFi connecting");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected");

    startCameraServer();

    Serial.print("Camera Ready! Use 'http://");
    Serial.print(WiFi.localIP());
    Serial.println("' to connect");

    Serial.println("\n[FOMO] Human Detection model loaded.");
    Serial.printf("[FOMO] Input: %dx%d px | Threshold: %.2f\n",
                  EI_CLASSIFIER_INPUT_WIDTH,
                  EI_CLASSIFIER_INPUT_HEIGHT,
                  (float)EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD);
}

// ===========================
// Arduino loop
// ===========================
void loop() {
    // Run FOMO inference every 3 seconds.
    // The web server stream runs in its own FreeRTOS task so this won't block it.
    run_fomo_inference();
    delay(3000);
}

// Guard: make sure the library was compiled for a camera sensor
#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Invalid model for current sensor — ensure you selected a Camera project in Edge Impulse."
#endif

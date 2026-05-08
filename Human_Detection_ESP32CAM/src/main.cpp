/* ============================================================
   ESP32-S3 N16R8 + OV2640 + FOMO Human Detection
   Project: CE103-Project (VinhNguyen)
   Label: "hum" | Input: 96x96 | Threshold: 0.5
   LED GPIO2 sáng khi phát hiện người
   ============================================================ */

#include <CE103-Project_inferencing.h>
#include "esp_camera.h"
#include "img_converters.h"
#include <Arduino.h>
// ============================================================
// PINOUT Camera OV2640 — ESP32-S3 N16R8 FPC connector
// ============================================================
#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   15
#define SIOD_GPIO_NUM   4
#define SIOC_GPIO_NUM   5
#define Y9_GPIO_NUM     16
#define Y8_GPIO_NUM     17
#define Y7_GPIO_NUM     18
#define Y6_GPIO_NUM     12
#define Y5_GPIO_NUM     10
#define Y4_GPIO_NUM     8
#define Y3_GPIO_NUM     9
#define Y2_GPIO_NUM     11
#define VSYNC_GPIO_NUM  6
#define HREF_GPIO_NUM   7
#define PCLK_GPIO_NUM   13

// ============================================================
// LED báo hiệu
// ============================================================
#define LED_PIN 2

// Buffer ảnh RGB
static uint8_t *rgb_buf = nullptr;

// ============================================================
// Callback lấy pixel cho Edge Impulse
// ============================================================
static int get_image_data(size_t offset, size_t length, float *out_ptr) {
    size_t px = offset * 3;
    for (size_t i = 0; i < length; i++) {
        // Đóng gói RGB888 thành 1 float (format Edge Impulse yêu cầu)
        out_ptr[i] = (rgb_buf[px + 2] << 16)
                   | (rgb_buf[px + 1] << 8)
                   |  rgb_buf[px];
        px += 3;
    }
    return 0;
}

// ============================================================
// Khởi động camera
// ============================================================
bool init_camera() {
    camera_config_t cfg;
    cfg.ledc_channel = LEDC_CHANNEL_0;
    cfg.ledc_timer   = LEDC_TIMER_0;
    cfg.pin_d0       = Y2_GPIO_NUM;
    cfg.pin_d1       = Y3_GPIO_NUM;
    cfg.pin_d2       = Y4_GPIO_NUM;
    cfg.pin_d3       = Y5_GPIO_NUM;
    cfg.pin_d4       = Y6_GPIO_NUM;
    cfg.pin_d5       = Y7_GPIO_NUM;
    cfg.pin_d6       = Y8_GPIO_NUM;
    cfg.pin_d7       = Y9_GPIO_NUM;
    cfg.pin_xclk     = XCLK_GPIO_NUM;
    cfg.pin_pclk     = PCLK_GPIO_NUM;
    cfg.pin_vsync    = VSYNC_GPIO_NUM;
    cfg.pin_href     = HREF_GPIO_NUM;
    cfg.pin_sccb_sda = SIOD_GPIO_NUM;
    cfg.pin_sccb_scl = SIOC_GPIO_NUM;
    cfg.pin_pwdn     = PWDN_GPIO_NUM;
    cfg.pin_reset    = RESET_GPIO_NUM;
    cfg.xclk_freq_hz = 20000000;
    cfg.pixel_format = PIXFORMAT_JPEG;
    cfg.frame_size   = FRAMESIZE_96X96;  // đúng với model 96x96
    cfg.jpeg_quality = 10;
    cfg.fb_count     = 1;
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

    if (esp_camera_init(&cfg) != ESP_OK) {
        Serial.println("❌ Camera init FAILED!");
        return false;
    }

    // Tối ưu sensor
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_framesize(s, FRAMESIZE_96X96);
        s->set_whitebal(s, 1);
        s->set_awb_gain(s, 1);
        s->set_exposure_ctrl(s, 1);
        s->set_aec2(s, 1);
        s->set_gainceiling(s, GAINCEILING_4X);
    }

    Serial.println("✅ Camera OK!");
    return true;
}

// ============================================================
// Chụp ảnh + chạy FOMO inference
// ============================================================
void run_detection() {
    // Chụp frame
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("⚠️ Không lấy được frame!");
        return;
    }

    // Cấp phát buffer RGB trong PSRAM
    size_t buf_size = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * 3;
    rgb_buf = (uint8_t *) ps_malloc(buf_size);
    if (!rgb_buf) {
        Serial.println("⚠️ Hết PSRAM!");
        esp_camera_fb_return(fb);
        return;
    }

    // Convert JPEG → RGB888
    bool ok = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, rgb_buf);
    esp_camera_fb_return(fb);

    if (!ok) {
        Serial.println("⚠️ Convert ảnh thất bại!");
        free(rgb_buf); rgb_buf = nullptr;
        return;
    }

    // Tạo signal cho Edge Impulse
    ei::signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data     = &get_image_data;

    // Chạy FOMO classifier
    ei_impulse_result_t result = {0};
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);

    free(rgb_buf); rgb_buf = nullptr;

    if (err != EI_IMPULSE_OK) {
        Serial.printf("⚠️ Lỗi inference: %d\n", err);
        return;
    }

    // ============================================================
    // Đọc kết quả bounding boxes FOMO
    // ============================================================
    bool detected = false;
    uint8_t count = 0;

    for (size_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value >= EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD) {
            count++;
            detected = true;
            Serial.printf("  [%d] label=%s | conf=%.0f%% | x=%u y=%u %ux%u\n",
                count, bb.label,
                bb.value * 100,
                bb.x, bb.y, bb.width, bb.height);
        }
    }

    if (detected) {
        Serial.printf("✅ Phát hiện %d người! (%d ms)\n", count, result.timing.classification);
        digitalWrite(LED_PIN, HIGH);
    } else {
        Serial.printf("🔍 Không có người (%d ms)\n", result.timing.classification);
        digitalWrite(LED_PIN, LOW);
    }
}

// ============================================================
// SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("====================================");
    Serial.println("  CE103 FOMO Human Detection");
    Serial.println("  ESP32-S3 N16R8 + OV2640");
    Serial.println("====================================");

    pinMode(LED_PIN, OUTPUT);

    // Nháy LED 3 lần báo boot
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, HIGH); delay(150);
        digitalWrite(LED_PIN, LOW);  delay(150);
    }

    if (!init_camera()) {
        // Nháy nhanh báo lỗi camera
        Serial.println("HALT: camera error");
        while (true) {
            digitalWrite(LED_PIN, HIGH); delay(80);
            digitalWrite(LED_PIN, LOW);  delay(80);
        }
    }

    Serial.printf("Model input: %dx%d\n", EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);
    Serial.printf("Label: hum | Threshold: %.0f%%\n", EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD * 100);
    Serial.println("Bắt đầu detection...\n");
}

// ============================================================
// LOOP
// ============================================================
void loop() {
    if ()

    run_detection();
    delay(100); // ~10 FPS
}

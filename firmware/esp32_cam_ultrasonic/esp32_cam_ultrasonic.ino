/*
 * Stray Dog Monitoring System — ESP32-CAM + HC-SR04 hardware node
 *
 * Streams MJPEG video over WiFi and serves ultrasonic distance readings
 * so the desktop app can watch a scene and measure how close a dog gets.
 *
 * Endpoints:
 *   http://<IP>:81/stream    - MJPEG video stream
 *   http://<IP>/distance     - JSON: {"distance_cm": 42.5, "timestamp": ...}
 *   http://<IP>/status       - JSON: device status (uptime, RSSI, IP, psram)
 *   http://<IP>/             - simple live dashboard
 *
 * Board: AI Thinker ESP32-CAM
 * Upload Speed: 115200
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ==================== CHANGE THESE ====================
const char* ssid     = "your-wifi";      // <-- your WiFi name
const char* password = "your-password";  // <-- your WiFi password
// ======================================================

// HC-SR04 pins
#define TRIG_PIN 14
#define ECHO_PIN 13

// AI Thinker ESP32-CAM pin map
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define FLASH_LED_PIN      4

WebServer server(80);
WiFiServer streamServer(81);

volatile float lastDistance = -1;
unsigned long lastDistanceTime = 0;

// ==================== ULTRASONIC ====================

float readDistance() {
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    long duration = pulseIn(ECHO_PIN, HIGH, 30000); // 30 ms timeout
    if (duration == 0) {
        return -1; // no echo (nothing in range)
    }
    float distance = duration * 0.0343 / 2.0; // cm
    if (distance > 400) {
        return -1; // out of range
    }
    return distance;
}

// ==================== CAMERA INIT ====================

bool initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode    = CAMERA_GRAB_LATEST;
    config.fb_location  = CAMERA_FB_IN_PSRAM;

    if (psramFound()) {
        config.frame_size   = FRAMESIZE_VGA;  // 640x480
        config.jpeg_quality = 12;
        config.fb_count     = 2;
    } else {
        config.frame_size   = FRAMESIZE_QVGA; // 320x240
        config.jpeg_quality = 15;
        config.fb_count     = 1;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return false;
    }

    sensor_t *s = esp_camera_sensor_get();
    s->set_brightness(s, 1);
    s->set_contrast(s, 1);
    s->set_saturation(s, 0);
    return true;
}

// ==================== HTTP HANDLERS ====================

void handleDistance() {
    float dist = readDistance();
    lastDistance = dist;
    lastDistanceTime = millis();

    String json = "{\"distance_cm\":";
    json += (dist < 0) ? "null" : String(dist, 1);
    json += ",\"timestamp\":" + String(millis()) + "}";

    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", json);
}

void handleStatus() {
    String json = "{";
    json += "\"device\":\"ESP32-CAM\",";
    json += "\"uptime_ms\":" + String(millis()) + ",";
    json += "\"wifi_rssi\":" + String(WiFi.RSSI()) + ",";
    json += "\"ip\":\"" + WiFi.localIP().toString() + "\",";
    json += "\"psram\":" + String(psramFound() ? "true" : "false") + ",";
    json += "\"last_distance\":";
    json += (lastDistance < 0) ? "null" : String(lastDistance, 1);
    json += "}";

    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", json);
}

void handleRoot() {
    String ip = WiFi.localIP().toString();
    String html = "<!DOCTYPE html><html><head><title>Stray Dog Monitoring - ESP32-CAM</title>";
    html += "<style>body{background:#1e1e1e;color:#fff;font-family:sans-serif;text-align:center;padding:20px}";
    html += "h1{color:#f59e0b}img{border:2px solid #333;border-radius:8px;max-width:100%}";
    html += ".info{background:#252525;padding:15px;border-radius:8px;margin:10px auto;max-width:640px}</style></head>";
    html += "<body><h1>Stray Dog Monitoring System</h1>";
    html += "<h3>ESP32-CAM Live Feed</h3>";
    html += "<img src='http://" + ip + ":81/stream' />";
    html += "<div class='info'>";
    html += "<p>Stream URL: <b>http://" + ip + ":81/stream</b></p>";
    html += "<p>Distance API: <b>http://" + ip + "/distance</b></p>";
    html += "<p id='dist'>Distance: reading...</p>";
    html += "</div>";
    html += "<script>setInterval(()=>{fetch('/distance').then(r=>r.json()).then(d=>{";
    html += "document.getElementById('dist').innerText='Distance: '+(d.distance_cm?d.distance_cm+' cm':'out of range');";
    html += "})},500);</script>";
    html += "</body></html>";
    server.send(200, "text/html", html);
}

// ==================== MJPEG STREAM ====================

void streamTask(void *pvParameters) {
    WiFiClient client;
    while (true) {
        client = streamServer.available();
        if (!client) {
            delay(10);
            continue;
        }
        String request = client.readStringUntil('\r');
        client.flush();

        client.println("HTTP/1.1 200 OK");
        client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
        client.println("Access-Control-Allow-Origin: *");
        client.println("Connection: close");
        client.println();

        while (client.connected()) {
            camera_fb_t *fb = esp_camera_fb_get();
            if (!fb) {
                delay(10);
                continue;
            }
            client.printf("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
            client.write(fb->buf, fb->len);
            client.println();
            esp_camera_fb_return(fb);
            delay(66); // ~15 FPS
        }
        client.stop();
    }
}

// ==================== SETUP & LOOP ====================

void setup() {
    Serial.begin(115200);
    Serial.println("\n\n=== Stray Dog Monitoring System - ESP32-CAM ===\n");

    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    pinMode(FLASH_LED_PIN, OUTPUT);
    digitalWrite(FLASH_LED_PIN, LOW);

    Serial.print("Initializing camera... ");
    if (!initCamera()) {
        Serial.println("FAILED! Restarting...");
        delay(2000);
        ESP.restart();
    }
    Serial.println("OK");

    Serial.printf("Connecting to WiFi: %s", ssid);
    WiFi.begin(ssid, password);
    WiFi.setSleep(false);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\nWiFi FAILED! Check SSID/password. Restarting...");
        delay(3000);
        ESP.restart();
    }

    Serial.println("\n========================================");
    Serial.println("  CONNECTED!");
    Serial.println("========================================");
    Serial.printf("  IP Address  : http://%s\n", WiFi.localIP().toString().c_str());
    Serial.printf("  Video Stream: http://%s:81/stream\n", WiFi.localIP().toString().c_str());
    Serial.printf("  Distance API: http://%s/distance\n", WiFi.localIP().toString().c_str());
    Serial.printf("  Status      : http://%s/status\n", WiFi.localIP().toString().c_str());
    Serial.println("========================================\n");

    for (int i = 0; i < 3; i++) {
        digitalWrite(FLASH_LED_PIN, HIGH);
        delay(100);
        digitalWrite(FLASH_LED_PIN, LOW);
        delay(100);
    }

    server.on("/", handleRoot);
    server.on("/distance", handleDistance);
    server.on("/status", handleStatus);
    server.begin();
    streamServer.begin();

    xTaskCreatePinnedToCore(streamTask, "stream", 8192, NULL, 1, NULL, 0);
    Serial.println("Ready! Open the IP address in your browser.\n");
}

void loop() {
    server.handleClient();
    static unsigned long lastRead = 0;
    if (millis() - lastRead > 200) {
        lastDistance = readDistance();
        lastRead = millis();
    }
}

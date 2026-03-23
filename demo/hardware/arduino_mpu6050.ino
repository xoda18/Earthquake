/*
 * arduino_mpu6050.ino
 *
 * MPU6050 3-axis accelerometer streaming to Serial.
 *
 * Hardware setup:
 *   MPU6050 VCC  → Arduino 5V  (or 3.3V)
 *   MPU6050 GND  → Arduino GND
 *   MPU6050 SDA  → Arduino A4  (SDA pin, or dedicated SDA)
 *   MPU6050 SCL  → Arduino A5  (SCL pin, or dedicated SCL)
 *
 * Serial output format (CSV):
 *   timestamp_ms,x_g,y_g,z_g
 *   0,0.01,0.02,9.81
 *   10,0.02,0.01,9.82
 *   ...
 *
 * Install MPU6050 library:
 *   Sketch → Include Library → Manage Libraries...
 *   Search "MPU6050" → Install "MPU6050 by Jeff Rowberg"
 */

#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"

MPU6050 mpu;
unsigned long start_time_ms;
unsigned int sample_count = 0;

// Accelerometer sensitivity: 16384 LSB/g (±2g range)
// Adjust if using different range (±4g: 8192, ±8g: 4096, ±16g: 2048)
const float ACCEL_SENSITIVITY = 16384.0;

void setup() {
  // Initialize I2C
  Wire.begin();
  Serial.begin(115200);

  // Initialize MPU6050
  mpu.initialize();

  // Verify connection
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed!");
    while (1);  // Hang if no sensor
  }

  // Optional: configure range, sample rate, etc.
  // mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);  // ±2g (default)
  // mpu.setRate(9);  // Sample rate divider (10 = 100 Hz at 1 kHz internal rate)

  start_time_ms = millis();
  Serial.println("timestamp_ms,x_g,y_g,z_g");  // CSV header
}

void loop() {
  // Read raw acceleration values (16-bit)
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);

  // Convert to g
  float x_g = ax / ACCEL_SENSITIVITY;
  float y_g = ay / ACCEL_SENSITIVITY;
  float z_g = az / ACCEL_SENSITIVITY;

  // Timestamp (milliseconds since start)
  unsigned long ts_ms = millis() - start_time_ms;

  // Print CSV row
  Serial.print(ts_ms);
  Serial.print(",");
  Serial.print(x_g, 3);  // 3 decimal places
  Serial.print(",");
  Serial.print(y_g, 3);
  Serial.print(",");
  Serial.println(z_g, 3);

  sample_count++;

  // Sampling interval: 10 ms = 100 Hz
  // Adjust for different sample rates (e.g., 5 ms = 200 Hz)
  delay(10);
}

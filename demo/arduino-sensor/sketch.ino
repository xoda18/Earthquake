/*
 * sketch.ino — MPU6500 raw I2C streaming
 *
 * Wiring:
 *   VCC → Arduino 3.3V
 *   GND → Arduino GND
 *   SDA → Arduino A4
 *   SCL → Arduino A5
 *   AD0 → GND (I2C address = 0x68)
 *
 * Output: CSV at 115200 baud
 *   timestamp_ms,ax,ay,az,gx,gy,gz
 */

#include <Wire.h>

#define MPU_ADDR      0x68
#define PWR_MGMT_1    0x6B
#define PWR_MGMT_2    0x6C
#define SIGNAL_RESET  0x68
#define ACCEL_XOUT_H  0x3B
#define ACCEL_CONFIG  0x1C
#define GYRO_CONFIG   0x1B
#define WHO_AM_I      0x75

const float ACCEL_SCALE = 16384.0;  // ±2g
const float GYRO_SCALE  = 131.0;    // ±250°/s

unsigned long start_time_ms;

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t readReg(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
  return Wire.read();
}

void setup() {
  Wire.begin();
  Serial.begin(115200);
  delay(200);

  writeReg(PWR_MGMT_1, 0x80);  // full reset
  delay(200);
  writeReg(PWR_MGMT_1, 0x00);  // wake up
  delay(100);
  writeReg(PWR_MGMT_2, 0x00);  // enable all axes
  delay(50);
  writeReg(SIGNAL_RESET, 0x07);
  delay(100);
  writeReg(ACCEL_CONFIG, 0x00);  // ±2g
  writeReg(GYRO_CONFIG,  0x00);  // ±250°/s
  delay(50);

  start_time_ms = millis();
  Serial.println("timestamp_ms,ax,ay,az,gx,gy,gz");
}

void loop() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14, (uint8_t)true);

  int16_t rax = (Wire.read() << 8) | Wire.read();
  int16_t ray = (Wire.read() << 8) | Wire.read();
  int16_t raz = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();  // skip temperature
  int16_t rgx = (Wire.read() << 8) | Wire.read();
  int16_t rgy = (Wire.read() << 8) | Wire.read();
  int16_t rgz = (Wire.read() << 8) | Wire.read();

  unsigned long ts = millis() - start_time_ms;

  Serial.print(ts);
  Serial.print(",");
  Serial.print(rax / ACCEL_SCALE, 3);
  Serial.print(",");
  Serial.print(ray / ACCEL_SCALE, 3);
  Serial.print(",");
  Serial.print(raz / ACCEL_SCALE, 3);
  Serial.print(",");
  Serial.print(rgx / GYRO_SCALE, 3);
  Serial.print(",");
  Serial.print(rgy / GYRO_SCALE, 3);
  Serial.print(",");
  Serial.println(rgz / GYRO_SCALE, 3);

  delay(10);  // 100 Hz
}

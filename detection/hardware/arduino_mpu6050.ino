/*
 * arduino_mpu6050.ino
 *
 * MPU6050/MPU6500 accelerometer streaming with I2C watchdog.
 * Auto-recovers from I2C bus hangs using hardware watchdog timer.
 *
 * Hardware:
 *   VCC → 3.3V, GND → GND, SDA → A4, SCL → A5
 *
 * Output: CSV at 115200 baud
 *   timestamp_ms,x_g,y_g,z_g
 */

#include <Wire.h>
#include <avr/wdt.h>  // hardware watchdog

#define MPU_ADDR      0x68
#define PWR_MGMT_1    0x6B
#define ACCEL_XOUT_H  0x3B
#define ACCEL_CONFIG  0x1C

const float ACCEL_SCALE = 16384.0;  // ±2g
unsigned long start_time_ms;
unsigned long last_good_read = 0;

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void initMPU() {
  Wire.begin();
  Wire.setClock(400000);  // 400kHz I2C

  writeReg(PWR_MGMT_1, 0x80);  // reset
  delay(200);
  writeReg(PWR_MGMT_1, 0x00);  // wake
  delay(100);
  writeReg(ACCEL_CONFIG, 0x00); // ±2g
  delay(50);
}

// Manual I2C recovery — toggles SCL to unstick SDA
void recoverI2C() {
  Wire.end();
  pinMode(A4, INPUT);    // release SDA
  pinMode(A5, OUTPUT);   // control SCL
  for (int i = 0; i < 16; i++) {
    digitalWrite(A5, LOW);
    delayMicroseconds(5);
    digitalWrite(A5, HIGH);
    delayMicroseconds(5);
  }
  pinMode(A5, INPUT);
  delay(10);
  initMPU();
}

void setup() {
  Serial.begin(115200);
  delay(200);

  initMPU();

  // Enable hardware watchdog — resets Arduino if loop() hangs for >4 sec
  wdt_enable(WDTO_4S);

  start_time_ms = millis();
  last_good_read = millis();
  Serial.println("timestamp_ms,x_g,y_g,z_g");
}

void loop() {
  wdt_reset();  // feed watchdog — prevents reset while running normally

  // If no good read for 2 seconds, try I2C recovery
  if (millis() - last_good_read > 2000) {
    recoverI2C();
    last_good_read = millis();
  }

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_XOUT_H);
  uint8_t err = Wire.endTransmission(false);

  if (err != 0) {
    // I2C error — try recovery
    recoverI2C();
    delay(10);
    return;
  }

  uint8_t count = Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)6, (uint8_t)true);
  if (count < 6) {
    recoverI2C();
    delay(10);
    return;
  }

  int16_t rax = (Wire.read() << 8) | Wire.read();
  int16_t ray = (Wire.read() << 8) | Wire.read();
  int16_t raz = (Wire.read() << 8) | Wire.read();

  // Check for garbage data (all 0xFF = -1)
  if (rax == -1 && ray == -1 && raz == -1) {
    recoverI2C();
    delay(10);
    return;
  }

  last_good_read = millis();
  unsigned long ts = millis() - start_time_ms;

  Serial.print(ts);
  Serial.print(",");
  Serial.print(rax / ACCEL_SCALE, 3);
  Serial.print(",");
  Serial.print(ray / ACCEL_SCALE, 3);
  Serial.print(",");
  Serial.println(raz / ACCEL_SCALE, 3);

  delay(10);  // 100 Hz
}

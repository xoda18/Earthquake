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

const int BUZZER_PIN = 13;
const int LED_PIN = 12;
const float ACCEL_SCALE = 16384.0;  // ±2g
unsigned long start_time_ms;
unsigned long last_good_read = 0;
bool alarm_active = false;
unsigned long alarm_start = 0;
const unsigned long ALARM_DURATION = 6000;  // 6 seconds of siren

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

  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  noTone(BUZZER_PIN);
  digitalWrite(LED_PIN, LOW);

  // Enable hardware watchdog — resets Arduino if loop() hangs for >4 sec
  wdt_enable(WDTO_4S);

  start_time_ms = millis();
  last_good_read = millis();
  Serial.println("timestamp_ms,x_g,y_g,z_g");
}

void loop() {
  wdt_reset();  // feed watchdog — prevents reset while running normally

  // Check for BUZZ command from Python
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "BUZZ") {
      alarm_active = true;
      alarm_start = millis();
    } else if (cmd == "STOP") {
      alarm_active = false;
      noTone(BUZZER_PIN);
      digitalWrite(LED_PIN, LOW);
    }
  }

  // Siren alarm — escalating woop-woop
  if (alarm_active) {
    unsigned long elapsed = millis() - alarm_start;
    if (elapsed >= ALARM_DURATION) {
      alarm_active = false;
      noTone(BUZZER_PIN);
      digitalWrite(LED_PIN, LOW);
    } else {
      // Phase repeats every 800ms: sweep 800Hz → 3000Hz then drop
      unsigned long phase = elapsed % 800;
      int freq;
      if (phase < 600) {
        // Rising sweep — peaks at 4kHz (loudest for small buzzers)
        freq = 2000 + (phase * 2000 / 600);
      } else {
        // Quick staccato drop
        freq = (phase % 100 < 50) ? 4000 : 800;
      }
      // Every other cycle goes higher for panic effect
      if ((elapsed / 800) % 2 == 1) {
        freq = freq * 5 / 4;  // +25% pitch = 5kHz peak
      }
      tone(BUZZER_PIN, freq);
      // LED blinks fast — 100ms on/off
      digitalWrite(LED_PIN, (elapsed / 100) % 2 == 0 ? HIGH : LOW);
    }
  }

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

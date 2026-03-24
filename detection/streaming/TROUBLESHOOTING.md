# Arduino Sensor Troubleshooting

## Quick Start

Every time you plug in the Arduino:
```bash
./demo/arduino-sensor/setup.sh
```

Then stream data:
```bash
source ../venv/bin/activate
python3 demo/arduino-sensor/stream.py --save sensor_data.csv
```

Analyze saved data:
```bash
python3 demo/detect_earthquake.py sensor_data.csv
```

---

## Check 1: Is the Arduino connected?

```bash
ls /dev/ttyACM* /dev/ttyUSB*
```

**Good:** `/dev/ttyACM0` appears
**Bad:** `No such file or directory`
**Fix:** Plug in the USB cable. Try a different USB port or cable.

---

## Check 2: Is it sending data?

```bash
sudo chmod 666 /dev/ttyACM0
stty -F /dev/ttyACM0 115200 raw -echo && timeout 5 cat /dev/ttyACM0
```

**Good:**
```
timestamp_ms,ax,ay,az,gx,gy,gz
2,0.056,-0.012,-1.025,-0.786,2.878,0.710
16,0.053,-0.014,-1.025,-0.779,2.718,0.817
```

**Bad (no output):** Sketch not uploaded → run `setup.sh`
**Bad (`connection failed`):** Sensor wiring issue → see Check 3

---

## Check 3: Is the sensor wired correctly?

Run the I2C scanner:
```bash
# Upload i2c_scanner (if available) or check wiring manually
```

**Correct wiring:**

| Sensor Pin | Arduino Pin | Notes |
|------------|-------------|-------|
| VCC        | 3.3V        | NOT 5V |
| GND        | GND         |  |
| SDA        | A4          | |
| SCL        | A5          | |
| AD0        | GND (on-board) | Sets address to 0x68 |

**I2C scanner finds `0x68`** → sensor is connected, sketch problem
**I2C scanner finds nothing** → wiring is loose or wrong

**Common wiring mistakes:**
- Using 5V instead of 3.3V (can damage sensor)
- SDA/SCL swapped
- Loose breadboard connection — push wires in firmly

---

## Check 4: Permission denied on serial port

```bash
sudo chmod 666 /dev/ttyACM0
```

Or add yourself to the serial group permanently:
```bash
sudo usermod -a -G uucp $USER
# then log out and log back in
```

---

## Check 5: Python can't import serial/numpy/etc.

Activate the venv first:
```bash
source ../venv/bin/activate
```

If packages are still missing:
```bash
pip install pyserial numpy pandas scipy matplotlib scikit-learn
```

---

## Sensor readings are all zero

The sensor is wired but not initializing. Run `setup.sh` to re-upload the sketch.
If zeros persist, the sensor may be damaged.

**Expected values when sensor is flat:**
- `ax` ≈ 0.05, `ay` ≈ -0.01, `az` ≈ -1.02 (gravity on Z axis)
- `|a|` ≈ 1.02g

---

## Arduino keeps resetting / upload fails

1. Try a different USB cable (data cable, not charge-only)
2. Check the Arduino power LED is on
3. Run with verbose output: `arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:uno /tmp/sketch -v`

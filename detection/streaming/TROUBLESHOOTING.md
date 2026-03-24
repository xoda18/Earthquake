# Arduino Sensor Troubleshooting

## Quick Start

```bash
# 1. Upload firmware (each time you reconnect Arduino USB)
./detection/streaming/setup.sh

# 2. Run LSTM detection in Docker
docker build -f detection/streaming/Dockerfile -t earthquake-detector .
docker run --rm --device /dev/ttyACM0 earthquake-detector --profile table
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
```

**Bad (no output):** Sketch not uploaded → run `./detection/streaming/setup.sh`
**Bad (`connection failed`):** Sensor wiring issue → see Check 3

---

## Check 3: Is the sensor wired correctly?

**Correct wiring (MPU6500):**

| Sensor Pin | Arduino Pin | Notes |
|------------|-------------|-------|
| VCC        | 3.3V        | NOT 5V — can damage sensor |
| GND        | GND         | |
| SDA        | A4          | I2C data |
| SCL        | A5          | I2C clock |
| AD0        | GND (on-board) | Sets address to 0x68 |

**Quick I2C test:** Upload an I2C scanner sketch. It should find device at `0x68`.
- Found `0x68` → sensor OK, sketch problem → re-upload
- Found nothing → wiring is loose or wrong

**Common mistakes:**
- 5V instead of 3.3V
- SDA/SCL swapped
- Loose breadboard connection — push wires in firmly

---

## Check 4: Permission denied on serial port

```bash
sudo chmod 666 /dev/ttyACM0
```

Or permanent fix:
```bash
sudo usermod -a -G uucp $USER
# log out and back in
```

---

## Check 5: Sensor readings are all zero

The sensor is wired but not initializing. Run `setup.sh` to re-upload the sketch.

**Expected when flat and still:**
- `ax` ≈ 0.06, `ay` ≈ -0.01, `az` ≈ -1.02
- `|a|` ≈ 1.02g

---

## Check 6: Docker says "No serial port"

The container needs `--device`:
```bash
docker run --rm --device /dev/ttyACM0 earthquake-detector
```

If port permissions error inside Docker:
```bash
sudo chmod 666 /dev/ttyACM0
```

---

## Check 7: Docker lag / delayed output

TensorFlow inference takes ~100-200ms per call. Reduce rate:
```bash
docker run --rm --device /dev/ttyACM0 earthquake-detector --rate 1
```

---

## Check 8: Model always says 0% or 100%

This is expected with the current LSTM model — it was trained on clean synthetic data so its outputs are very confident. Use `--threshold` to control when EARTHQUAKE triggers:
```bash
# Only trigger on 100% confidence
docker run --rm --device /dev/ttyACM0 earthquake-detector --threshold 0.95

# Use table profile with gain amplification
docker run --rm --device /dev/ttyACM0 earthquake-detector --profile table
```

---

## Rebuilding Docker after code changes

```bash
docker build -f detection/streaming/Dockerfile -t earthquake-detector .
```

import sys, time
import serial

# set DTR/RTS false BEFORE open, else the auto-reset circuit reboots the board
port = serial.Serial()
port.port = "COM5"
port.baudrate = 115200
port.timeout = 1
port.rts = False
port.dtr = False
port.open()
end = time.time() + 35
while time.time() < end:
    data = port.read(4096)
    if data:
        sys.stdout.buffer.write(data.decode("ascii", "replace").encode("ascii", "replace"))
        sys.stdout.flush()
port.close()

import serial
from serial.tools import list_ports
import time
import sys

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
# IMPORTANT: make sure this matches your Arduino's serial port
# Mac/Linux: '/dev/tty.usbmodem...' or '/dev/ttyUSB0'
SERIAL_PORT = 'COM7' 
BAUD_RATE = 115200

def print_menu():
    print("\n--- RPS Robotic Arm Control ---")
    print("1: Rock")
    print("2: Paper")
    print("3: Scissors")
    print("0: Paper / Neutral")
    print("q: Quit")
    print("-------------------------------")

def main():
    try:
        # show available ports for debugging
        ports = list(list_ports.comports())
        print("Available serial ports:")
        if ports:
            for p in ports:
                print(f" - {p.device} | {p.description} | {p.hwid}")
        else:
            print(" - (none found)")

        # Connect to port
        print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for Arduino reset (DTR reset)
        print(f"Connected successfully on {ser.port} (open={ser.is_open})")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        print("Hint: Check the port name and close any other programs using it (like Arduino IDE).")
        sys.exit(1)

    print_menu()

    while True:
        user_input = input("Enter command > ").strip().lower()

        command = None
        action_name = ""

        if user_input == '1':
            command = b'\x01' # OP_ROCK
            action_name = "ROCK"
        elif user_input == '2':
            command = b'\x02' # OP_PAPER
            action_name = "PAPER"
        elif user_input == '3':
            command = b'\x03' # OP_SCISSORS
            action_name = "SCISSORS"
        elif user_input == '0':
            command = b'\x00' # OP_NEUTRAL
            action_name = "PAPER / NEUTRAL"
        elif user_input == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid command. Try 1, 2, 3 or 0.")
            continue

        if command:
            try:
                print(f"[DEBUG] Writing to {SERIAL_PORT}: {command!r}")
                ser.write(command)
                ser.flush()
                print(f"Sent: {action_name}")
                # small delay to let Arduino process and respond
                time.sleep(0.1)
                # read all available lines from Arduino
                while ser.in_waiting:
                    try:
                        line = ser.readline().decode('utf-8', errors='replace').strip()
                        if line:
                            print(f"Arduino says: {line}")
                    except Exception as e:
                        print(f"[DEBUG] Read error: {e}")
                        break
            except serial.SerialTimeoutException:
                print("Error: Write timeout.")
            except Exception as e:
                print(f"Communication error: {e}")

    ser.close()
    print("Connection closed.")

if __name__ == "__main__":
    main()
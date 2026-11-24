# Adapted from the station LeRobot build and transport checks for PressB.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real libfranka move/stop concurrency against localhost, with no hardware.

The fake server deliberately withholds motion completion until stop arrives,
then varies reply order and packet fragmentation. A subprocess timeout catches
deadlocks. This validates SDK transport, not physical gripper stopping latency.
"""

import argparse
import json
import socket
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

HEADER = struct.Struct("<HII")
ITERATIONS = 80
CLIENT = r"""
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <franka/exception.h>
#include <franka/gripper.h>

int main() {
  try {
    franka::Gripper hand("127.0.0.1");
    for (int i = 0; i < 80; ++i) {
      auto motion = std::async(std::launch::async, [&hand, i]() {
        try {
          if (i % 2) hand.grasp(0.0, 0.1, 20.0, 0.005, 0.08);
          else hand.move(0.02, 0.1);
          throw std::runtime_error("motion unexpectedly returned before cancellation");
        } catch (const franka::CommandException& error) {
          if (std::string(error.what()).find("aborted") == std::string::npos) throw;
        }
      });
      // Server never completes motion on its own; stop must be able to enter
      // while another thread is inside tcpBlockingReceiveResponse.
      std::this_thread::sleep_for(std::chrono::milliseconds(15));
      auto stop = std::async(std::launch::async, [&hand]() { return hand.stop(); });
      if (!stop.get()) throw std::runtime_error("stop returned false");
      motion.get();
      if (!hand.move(0.06, 0.1)) throw std::runtime_error("next motion failed");
    }
    std::cout << "80 concurrent stop cycles and 80 subsequent moves passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""


def receive_exact(connection: socket.socket, size: int) -> bytes:
    """Receive one bounded packet fragment, rejecting early EOF."""
    value = bytearray()
    while len(value) < size:
        chunk = connection.recv(size - len(value))
        if not chunk:
            raise EOFError("Native test client disconnected")
        value.extend(chunk)
    return bytes(value)


def receive_command(connection: socket.socket) -> tuple[int, int, bytes]:
    """Parse the packed protocol header and its request payload."""
    command, identifier, size = HEADER.unpack(receive_exact(connection, HEADER.size))
    if not HEADER.size <= size <= 100:
        raise ValueError(f"Invalid command packet size {size}")
    return command, identifier, receive_exact(connection, size - HEADER.size)


def response(command: int, identifier: int, status: int) -> bytes:
    """Encode one move, grasp or stop completion status."""
    return HEADER.pack(command, identifier, HEADER.size + 2) + struct.pack("<H", status)


def run_server(listener: socket.socket, errors: list[str], metrics: list[float]) -> None:
    """Emulate only the gripper protocol on an explicitly local connection."""
    phase = "accept"
    try:
        connection, address = listener.accept()
        if address[0] != "127.0.0.1":
            raise ValueError("Unexpected non-local client")
        with connection:
            connection.settimeout(3)
            phase = "connect handshake"
            command, identifier, payload = receive_command(connection)
            if command != 0 or struct.unpack("<HH", payload)[0] != 3:
                raise ValueError("Incorrect gripper connect handshake")
            connection.sendall(HEADER.pack(0, identifier, HEADER.size + 4) + struct.pack("<HH", 0, 3))
            for index in range(ITERATIONS):
                phase = f"cycle {index}: first command"
                first = receive_command(connection)
                started = time.monotonic()
                phase = f"cycle {index}: second command after {first[0]}"
                second = receive_command(connection)
                metrics.append((time.monotonic() - started) * 1000)
                # Scheduling may issue stop before the motion request. Both
                # must arrive without either command having received a reply.
                received = {first[0]: first, second[0]: second}
                motion_type = 2 if index % 2 else 3
                if set(received) != {motion_type, 4}:
                    raise ValueError(f"Unexpected concurrent requests: {received}")
                replies = [
                    response(motion_type, received[motion_type][1], 3),
                    response(4, received[4][1], 0),
                ]
                if index % 2:
                    replies.reverse()
                packet = b"".join(replies)
                if index % 3 == 0:
                    # Split within a header to exercise shared pending buffer.
                    connection.sendall(packet[:5])
                    time.sleep(0.001)
                    connection.sendall(packet[5:])
                else:
                    connection.sendall(packet)
                phase = f"cycle {index}: follow-up move after concurrent replies"
                command, identifier, payload = receive_command(connection)
                if command != 3 or struct.unpack("<dd", payload) != (0.06, 0.1):
                    raise ValueError("Follow-up move was lost or corrupted")
                connection.sendall(response(command, identifier, 0))
    except (OSError, EOFError, ValueError, struct.error) as error:
        errors.append(f"{phase}: {type(error).__name__}: {error}")


def main() -> None:
    """Compile the client and verify concurrent transport against localhost."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-build", type=Path, required=True, help="Private build/sdk directory")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sdk = args.sdk_build.resolve()
    source = sdk.parents[1] / "source"
    library = sdk / "libpressb_franka.so"
    if not library.is_file() or not (sdk.parents[1] / "pressb_private_sdk.json").is_file():
        raise ValueError("Expected managed private Franka SDK build")
    with tempfile.TemporaryDirectory(prefix="franka-native-stop-") as directory:
        temporary = Path(directory)
        cpp = temporary / "client.cpp"
        executable = temporary / "client"
        cpp.write_text(CLIENT)
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O2",
                "-pthread",
                str(cpp),
                f"-I{source / 'include'}",
                f"-L{sdk}",
                f"-Wl,-rpath,{sdk}",
                "-lpressb_franka",
                "-o",
                str(executable),
            ],
            check=True,
        )
        linkage = subprocess.run(["ldd", str(executable)], check=True, text=True, capture_output=True).stdout
        if str(library.parent) not in linkage:
            raise RuntimeError("Test executable did not resolve the private SDK")
        errors: list[str] = []
        intervals: list[float] = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 1338))
            listener.listen(1)
            listener.settimeout(5)
            server = threading.Thread(target=run_server, args=(listener, errors, intervals), daemon=True)
            server.start()
            result = subprocess.run(
                [str(executable)],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            server.join(timeout=4)
        if server.is_alive() or errors or result.returncode:
            raise RuntimeError(
                f"Native concurrency regression failed: {errors}, {result.stdout}, {result.stderr}"
            )
        if len(intervals) != ITERATIONS:
            raise RuntimeError("Native test ended before all cycles completed")
        report = {
            "cycles": ITERATIONS,
            "following_moves": ITERATIONS,
            "physical_hardware_connected": False,
            "maximum_request_arrival_gap_ms": max(intervals),
            "stdout": result.stdout.strip(),
            "ldd": linkage,
            "scope": "SDK concurrency on loopback; not a physical stop latency measurement",
        }
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

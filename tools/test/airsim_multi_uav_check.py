"""Quick health check for AirSim multi-UAV setup.

Usage:
  python tools/test/airsim_multi_uav_check.py --vehicles Drone1 Drone2
"""

import argparse
import sys
import socket
import platform
import subprocess
import time
import airsim


def scan_open_ports(host: str, ports, timeout_s: float):
    open_ports = []
    for p in ports:
        try:
            with socket.create_connection((host, p), timeout=timeout_s):
                open_ports.append(p)
        except OSError:
            continue
    return open_ports


def get_port_owner_hint(host: str, port: int):
    """Best-effort port owner hint.
    Returns a short string or None.
    """
    try:
        if platform.system().lower().startswith("win"):
            out = subprocess.check_output(["netstat", "-ano"], text=True, stderr=subprocess.STDOUT)
            for line in out.splitlines():
                if f"{host}:{port}" in line or f":{port}" in line:
                    line = line.strip()
                    if "LISTENING" in line or "ESTABLISHED" in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            return f"Port {port} appears in netstat (PID={pid})."
        else:
            out = subprocess.check_output(["ss", "-ltnp"], text=True, stderr=subprocess.STDOUT)
            for line in out.splitlines():
                if f":{port}" in line:
                    return f"Port {port} listening entry: {line.strip()}"
    except Exception:
        return None
    return None


def check_vehicle(client, vehicle_name: str) -> bool:
    ok = True
    print(f"\n[Check] {vehicle_name}")

    # API control + arm
    try:
        client.enableApiControl(True, vehicle_name=vehicle_name)
        client.armDisarm(True, vehicle_name=vehicle_name)
        print("  - API control/arm: OK")
    except Exception as exc:  # noqa: BLE001
        print(f"  - API control/arm: FAIL ({exc})")
        return False

    # Pose
    try:
        pose = client.simGetVehiclePose(vehicle_name=vehicle_name)
        print(
            "  - Pose: "
            f"x={pose.position.x_val:.2f}, y={pose.position.y_val:.2f}, z={pose.position.z_val:.2f}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  - Pose: FAIL ({exc})")
        ok = False

    # State
    try:
        state = client.getMultirotorState(vehicle_name=vehicle_name)
        landed = state.landed_state
        print(f"  - MultirotorState: OK (landed_state={landed})")
    except Exception as exc:  # noqa: BLE001
        print(f"  - MultirotorState: FAIL ({exc})")
        ok = False

    # Image
    try:
        responses = client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)],
            vehicle_name=vehicle_name,
        )
        if not responses or responses[0].width == 0:
            print("  - Depth image: FAIL (empty response)")
            ok = False
        else:
            print(f"  - Depth image: OK ({responses[0].width}x{responses[0].height})")
    except TypeError:
        # Fallback for older AirSim API signature
        try:
            responses = client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)]
            )
            if not responses or responses[0].width == 0:
                print("  - Depth image: FAIL (empty response/fallback)")
                ok = False
            else:
                print(
                    "  - Depth image: OK via fallback signature "
                    f"({responses[0].width}x{responses[0].height})"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  - Depth image: FAIL ({exc})")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"  - Depth image: FAIL ({exc})")
        ok = False

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="AirSim multi-UAV quick check")
    parser.add_argument(
        "--vehicles",
        nargs="+",
        default=["Drone1", "Drone2"],
        help="Vehicle names to check (default: Drone1 Drone2)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="AirSim RPC host")
    parser.add_argument("--port", type=int, default=41451, help="AirSim RPC port")
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=3.0,
        help="Socket timeout (seconds) before failing fast",
    )
    parser.add_argument(
        "--handshake-retries",
        type=int,
        default=5,
        help="Number of confirmConnection retries before declaring failure",
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=2.0,
        help="Seconds to wait between handshake retries",
    )
    args = parser.parse_args()

    print(
        f"Connecting to AirSim RPC {args.host}:{args.port} "
        f"(timeout={args.connect_timeout:.1f}s)..."
    )

    # Fast-fail check to avoid confirmConnection hanging indefinitely.
    try:
        with socket.create_connection((args.host, args.port), timeout=args.connect_timeout):
            pass
    except OSError as exc:
        print(f"Connection failed before AirSim handshake: {exc}")
        print("Hint: start UE/AirSim scene first and verify RPC is enabled.")
        return 2

    client = airsim.MultirotorClient(ip=args.host, port=args.port, timeout_value=args.connect_timeout)
    last_exc = None
    for i in range(args.handshake_retries):
        try:
            if i > 0:
                print(f"Retrying AirSim handshake ({i+1}/{args.handshake_retries})...")
            client.confirmConnection()
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if i < args.handshake_retries - 1:
                time.sleep(args.retry_interval)

    if last_exc is not None:
        print(f"Connection failed: {last_exc}")
    try:
        client.confirmConnection()
    except Exception as exc:  # noqa: BLE001
        print(f"Connection failed: {exc}")
        print("Hint: check SimMode=Multirotor and whether RPC server is listening.")
        common_ports = list(range(41451, 41461))
        open_ports = scan_open_ports(args.host, common_ports, timeout_s=0.2)
        if open_ports:
            print(f"Detected open local ports in [41451-41460]: {open_ports}")
            if args.port not in open_ports:
                print(
                    "Your configured --port is not open. "
                    "Try one of the detected ports and re-run the checker."
                )
            else:
                print(
                    "Configured port is open but AirSim handshake timed out. "
                    "Usually this means the service on that port is not AirSim yet, "
                    "or UE scene has not fully loaded."
                )
                owner_hint = get_port_owner_hint(args.host, args.port)
                if owner_hint:
                    print(owner_hint)
        else:
            print("No common AirSim RPC ports are open on localhost.")
            print("Please start UE/AirSim scene first and wait until world fully loads.")
        return 2

    all_ok = True
    for vehicle_name in args.vehicles:
        if not check_vehicle(client, vehicle_name):
            all_ok = False

    if all_ok:
        print("\n[Result] PASS: all requested vehicles are healthy.")
        return 0

    print("\n[Result] FAIL: at least one vehicle check failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

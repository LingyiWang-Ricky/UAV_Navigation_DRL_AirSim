"""Quick health check for AirSim multi-UAV setup.

Usage:
  python tools/test/airsim_multi_uav_check.py --vehicles Drone1 Drone2
"""

import argparse
import sys
import socket
import airsim


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
    try:
        client.confirmConnection()
    except Exception as exc:  # noqa: BLE001
        print(f"Connection failed: {exc}")
        print("Hint: check SimMode=Multirotor and whether RPC server is listening.")
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

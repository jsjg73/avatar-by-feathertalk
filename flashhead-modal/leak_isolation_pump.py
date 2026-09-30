"""Client for leak_isolation_server.py: hits /speak repeatedly over real HTTP
(same UploadFile path as production), waits for the queue to fully drain
between checkpoints, and reports the server process's RSS at each point.

Usage: python leak_isolation_pump.py <server_pid> [--calls 100] [--wav test_line.wav]
"""
import argparse
import time
import urllib.request


def rss_kb(pid: int) -> int:
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1])
    return -1


def speak(wav_path: str) -> None:
    with open(wav_path, "rb") as f:
        data = f.read()
    boundary = "----X"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=file; filename=t.wav\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8999/speak", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    urllib.request.urlopen(req).read()


def pending() -> int:
    import json
    with urllib.request.urlopen("http://127.0.0.1:8999/status") as r:
        return json.load(r)["pending"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("server_pid", type=int)
    p.add_argument("--calls", type=int, default=100)
    p.add_argument("--wav", default="test_line.wav")
    p.add_argument("--batch", type=int, default=20)
    args = p.parse_args()

    print("rss before:", rss_kb(args.server_pid))
    done = 0
    while done < args.calls:
        n = min(args.batch, args.calls - done)
        for _ in range(n):
            speak(args.wav)
        done += n
        print(f"rss after {done} calls (pending={pending()}):", rss_kb(args.server_pid))

    print("draining...")
    while pending() > 0:
        time.sleep(1)
    print("rss after full drain:", rss_kb(args.server_pid))


if __name__ == "__main__":
    main()

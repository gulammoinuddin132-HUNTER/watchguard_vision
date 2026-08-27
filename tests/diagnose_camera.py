"""
Detailed Camera & OpenCV Diagnostic Script
-------------------------------------------
1. Checks active processes for possible webcam locks.
2. Checks Windows webcam privacy registry settings.
3. Systematically tests OpenCV VideoCapture on indexes 0-5 across all backends:
   - CAP_ANY
   - CAP_DSHOW
   - CAP_MSMF
   - CAP_OBSENSOR
   - CAP_FFMPEG
"""

import os
import subprocess
import sys
import winreg
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2


def check_running_processes():
    print("=== [1] Running Process Inspection ===")
    try:
        output = subprocess.check_output("tasklist /V /FO CSV", shell=True, text=True, errors="ignore")
        lines = output.strip().split("\n")
        suspicious = []
        for line in lines[1:]:
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) >= 9:
                proc_name = parts[0].lower()
                window_title = parts[8].lower()
                if any(k in proc_name or k in window_title for k in ["camera", "zoom", "teams", "skype", "obs", "chrome", "edge", "discord", "meet"]):
                    suspicious.append(f"PID: {parts[1]:<6} | Process: {parts[0]:<20} | Title: {parts[8]}")
        
        if suspicious:
            print(f"Found {len(suspicious)} potential camera-using processes:")
            for s in suspicious:
                print("  -> " + s)
        else:
            print("No obvious camera-locking application found.")
    except Exception as e:
        print("Error checking processes:", e)


def check_camera_privacy():
    print("\n=== [2] Windows Camera Access & Privacy Settings ===")
    try:
        key_path = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            val, _ = winreg.QueryValueEx(key, "Value")
            print(f"User Webcam Capability Access: {val} (Allow = app access enabled)")
    except Exception as e:
        print("Could not query HKCU webcam consent:", e)

    try:
        key_path_global = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path_global) as key:
            val, _ = winreg.QueryValueEx(key, "Value")
            print(f"System Global Webcam Capability Access: {val}")
    except Exception as e:
        print("Could not query HKLM webcam consent:", e)


def test_opencv_camera_matrix():
    print("\n=== [3] OpenCV VideoCapture Multi-Backend Test (Indexes 0 through 5) ===")
    print(f"OpenCV Version: {cv2.__version__}")
    
    backends = [
        ("CAP_ANY", cv2.CAP_ANY),
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("CAP_MSMF", cv2.CAP_MSMF),
        ("CAP_OBSENSOR", getattr(cv2, "CAP_OBSENSOR", 2600)),
        ("CAP_FFMPEG", cv2.CAP_FFMPEG),
    ]

    results = []

    print("-" * 80)
    print(f"{'Index':<6} | {'Backend':<14} | {'isOpened()':<10} | {'Frame Read':<10} | {'Resolution / Details':<30}")
    print("-" * 80)

    for idx in range(6):
        for b_name, b_flag in backends:
            is_opened = False
            frame_read = False
            details = "N/A"
            try:
                cap = cv2.VideoCapture(idx, b_flag)
                is_opened = cap.isOpened()
                if is_opened:
                    ret, frame = cap.read()
                    frame_read = bool(ret and frame is not None and frame.size > 0)
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    if frame_read:
                        details = f"SUCCESS: {w}x{h} @ {fps:.1f} FPS (Shape: {frame.shape})"
                    else:
                        details = f"Opened, but frame read returned False ({w}x{h})"
                    cap.release()
                else:
                    details = "Failed to open device"
            except Exception as e:
                details = f"Exception: {str(e)[:40]}"

            res_entry = {
                "index": idx,
                "backend": b_name,
                "is_opened": is_opened,
                "frame_read": frame_read,
                "details": details,
            }
            results.append(res_entry)
            print(f"{idx:<6} | {b_name:<14} | {str(is_opened):<10} | {str(frame_read):<10} | {details:<30}")

    print("-" * 80)

    working = [r for r in results if r["frame_read"]]
    print(f"\nTotal Working Configurations Found: {len(working)}")
    for w in working:
        print(f"  ⭐ SUCCESS: Index {w['index']} with {w['backend']} -> {w['details']}")

    return results


if __name__ == "__main__":
    check_running_processes()
    check_camera_privacy()
    test_opencv_camera_matrix()

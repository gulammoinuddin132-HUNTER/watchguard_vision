"""
Query Camera Devices via PowerShell
"""
import subprocess

cmd = ['powershell', '-NoProfile', '-Command', 'Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq "Camera" } | Format-List Name, DeviceID, Status, Present, Service']
out = subprocess.check_output(cmd, text=True, errors='ignore')
print(out)

"""
Check HP Services and Camera State
"""
import subprocess

cmd = ['powershell', '-NoProfile', '-Command', 'Get-Process | Where-Object { $_.ProcessName -like "*hp*" -or $_.ProcessName -like "*omen*" } | Select-Object Id, ProcessName, MainWindowTitle']
out = subprocess.check_output(cmd, text=True, errors='ignore')
print(out)

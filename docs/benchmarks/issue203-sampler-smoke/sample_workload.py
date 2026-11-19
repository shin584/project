import subprocess
import sys
import time
payload = bytearray(8 * 1024 * 1024)
child = subprocess.Popen([sys.executable, '-c', 'import time; payload=bytearray(24*1024*1024); time.sleep(2)'])
child.wait()
time.sleep(.5)

import os
import sys

print("PYTHONPATH Environment Variable:")
print(os.getenv("PYTHONPATH"))
print("\nPython's sys.path:")
for path in sys.path:
    print(path)
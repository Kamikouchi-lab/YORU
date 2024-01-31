import sys

# for path in sys.path:
#     print(path)

sys.path.append("../yoru")

from yoru.app import main

print("Initializing YORU.....")
main()

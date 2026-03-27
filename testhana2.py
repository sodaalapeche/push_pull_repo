import ctypes
import os

# === Path to your 64-bit DCAM DLL ===
DCAM_DLL_PATH = r"C:\Program Files\Hamamatsu\DCAM-API\redist\win64\vc2015\dcamapi.dll"

if not os.path.exists(DCAM_DLL_PATH):
    raise FileNotFoundError(f"DCAM DLL not found at: {DCAM_DLL_PATH}")

# Load DLL
dcam = ctypes.WinDLL(DCAM_DLL_PATH)

# --- Error codes (partial) ---
DCAMERR_NOERROR = 0x00000000

# === DCAMAPI_INIT Struct ===
class DCAMAPI_INIT(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int32),
        ("iDeviceCount", ctypes.c_int32),
        ("reserved1", ctypes.c_int32),
        ("initoptionbytes", ctypes.c_void_p),
        ("initoption", ctypes.c_void_p)
    ]

# === DCAMDEV_OPEN Struct ===
class DCAMDEV_OPEN(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int32),
        ("index", ctypes.c_int32),
        ("handle", ctypes.c_void_p)
    ]

# === Set Function Signatures ===
dcam.dcamapi_init.argtypes = [ctypes.POINTER(DCAMAPI_INIT)]
dcam.dcamapi_init.restype = ctypes.c_int32

dcam.dcamdev_open.argtypes = [ctypes.POINTER(DCAMDEV_OPEN)]
dcam.dcamdev_open.restype = ctypes.c_int32

dcam.dcamdev_close.argtypes = [ctypes.c_void_p]
dcam.dcamdev_close.restype = ctypes.c_int32

dcam.dcamapi_uninit.argtypes = []
dcam.dcamapi_uninit.restype = ctypes.c_int32

# === Initialize DCAM API ===
init = DCAMAPI_INIT()
init.size = ctypes.sizeof(DCAMAPI_INIT)
init.initoptionbytes = None
init.initoption = None

err = dcam.dcamapi_init(ctypes.byref(init))
if err != DCAMERR_NOERROR:
    raise RuntimeError(f"dcamapi_init failed with error: {err:#X}")
print(f"✅ DCAM initialized. Devices found: {init.iDeviceCount}")

if init.iDeviceCount == 0:
    raise RuntimeError("No camera detected.")

# === Open first camera ===
open_dev = DCAMDEV_OPEN()
open_dev.size = ctypes.sizeof(DCAMDEV_OPEN)
open_dev.index = 0

err = dcam.dcamdev_open(ctypes.byref(open_dev))
if err != DCAMERR_NOERROR:
    raise RuntimeError(f"dcamdev_open failed: {err:#X}")

print(f"✅ Camera opened. Handle: {open_dev.handle}")

# === Close and clean up ===
dcam.dcamdev_close(open_dev.handle)
dcam.dcamapi_uninit()
print("✅ Clean shutdown complete.")

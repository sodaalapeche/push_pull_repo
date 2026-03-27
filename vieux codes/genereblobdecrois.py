import numpy as np
import os
from PIL import Image

n = 200
size = 1000
a = 30
D = 20
m0 = 10**10
root_folder = "/home/chorus/EXP_TO_TREAT/fauxblobgaussien/serie1/"

t_max = 100
times = np.linspace(0.1, t_max, n)

os.makedirs(root_folder, exist_ok=True)

x = np.linspace(-size // 2, size // 2, size)
y = np.linspace(-size // 2, size // 2, size)
X, Y = np.meshgrid(x, y)
R2 = X**2 + Y**2

s0 = a**2 + 2 * D * times[0]
C0 = (m0 / (2 * np.pi * s0)) * np.exp(-R2 / (2 * s0))
C_max_global = C0.max()
k = 65535 / C_max_global


for i, t in enumerate(times, 1):
    s = a**2 + 2 * D * t
    C = (m0 / (2 * np.pi * s)) * np.exp(-R2 / (2 * s))
    C_uint16 = np.clip(C * (60000 / C_max_global), 0, 65535).astype(np.uint16)
    img = Image.fromarray(C_uint16, mode="I;16")
    img.save(os.path.join(root_folder, f"{i}.tif"))

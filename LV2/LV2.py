import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.gridspec as gridspec


# ZADATAK 1 - CRTANJE GEOMETRIJSKOG OBLIKA

print("\n" + "="*60)
print("ZADATAK 1 - CRTANJE GEOMETRIJSKOG OBLIKA")
print("="*60)

x = np.array([1.0, 2.0, 3.0, 3.0, 1.0])

y = np.array([1.0, 2.0, 2.0, 1.0, 1.0])

plt.figure(figsize=(6, 5))
plt.plot(x, y,
        color='blue',
        linewidth=2, 
        marker='o',
        markersize=8,
        label='Trapez')

plt.fill(x, y, alpha=0.15, color='blue')
plt.axis([0.0, 4.0, 0.0, 4.0])
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Zadatak 1 - CRTANJE GEOMETRIJSKOG OBLIKA')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# ZADATAK 2 - ANALIZA mtcars DATASET-a
print("\n" + "="*60)
print("ZADATAK 2 - ANALIZA mtcars DATASET-a")
print("="*60)

data = np.loadtxt(
    open("mtcars.csv", "rb"),
    usecols=(1, 2, 3, 4, 5, 6),
    delimiter=",",
    skiprows=1
)

mpg = data[:, 0]
cyl = data[:, 1]
hp = data[:, 3]
wt = data[:, 5]

print(f"Ucitan dataset: {data.shape[0]} automobila, {data.shape[1]} varijabli.")

plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    hp, mpg, 
    s = wt * 50,
    c = wt,
    cmap = 'coolwarm',
    alpha = 0.7,
    edgecolors='black',
    linewidth = 0.5
)

cbar = plt.colorbar(scatter)
cbar.set_label('Težina wt (lb/1000)', fontsize=11)
plt.xlabel('Konjska snaga hp', fontsize=12)
plt.ylabel('Potrosnja (mpg)', fontsize=12)
plt.title('Ovisnost potrošnje o konjskoj snazi\n(veličina točke = težina vozila wt)', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

print("\nSTATISTIKE mpg - SVI AUTOMOBILI (n=32):")
print(f" Minimum: {np.min(mpg):.2f} mpg")
print(f" Maksimum: {np.max(mpg):.2f} mpg")
print(f" Srednja vrijednost: {np.mean(mpg):.2f} mpg")


maska_6 = (cyl == 6)
mpg_6 = mpg[maska_6]

print(f"\nSTATISTIKE mpg - AUTOMOBILI SA 6 CILINDARA (n={len(mpg_6)}):")
print(f" Minimum: {np.min(mpg_6):.2f} mpg")
print(f" Maksimum: {np.max(mpg_6):.2f} mpg")
print(f" Srednja vrijednost: {np.mean(mpg_6):.2f} mpg")


# ZADATAK 3 - MANIPULACIJA SLIKE tiger.png
print("\n" + "="*60)
print("ZADATAK 3 - MANIPULACIJA SLIKE tiger.png")
print("="*60)

img_pil = Image.open("tiger.png").convert('L')
img = np.array(img_pil, dtype=np.float32) / 255.0
H, W = img.shape
print(f"Ucitana slika: {img.shape}, dtype={img.dtype}, raspon=[{img.min():.2f}, {img.max():.2f} ] ")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)

def prikazi_subplot(ax, slika, naslov):
    ax.imshow(slika, cmap='gray', vmin=0.0, vmax=1.0)
    ax.set_title(naslov, fontsize=9, fontweight='bold')
    ax.axis('off')

# Originalna slika
ax = fig.add_subplot(gs[0, 0])
prikazi_subplot(ax, img, f'Originalna slika\n{img.shape}')

# a) Posvijetljenje
img_bright = np.clip(img + 0.3, 0.0, 1.0)
ax = fig.add_subplot(gs[0, 1])
prikazi_subplot(ax, img_bright, f'a) Posvijetljena slika (+0.3)\n{img_bright.shape}')
print("a) Posvijetljena slika: np.clip(img + 0.3, 0.0, 1.0)")

# b) Rotacija za 90 stupnjeva u smjeru kazaljke na satu
img_rot = np.rot90(img, k=-1)
ax = fig.add_subplot(gs[0, 2])
prikazi_subplot(ax, img_rot, f'b) Rotirana slika (90° CW)\n{img_rot.shape}')
print(f"b) Rotirana slika: {img.shape} -> {img_rot.shape}")

# c) Zrcaljenje (horizontalno)
img_mirror = np.fliplr(img)
ax = fig.add_subplot(gs[1, 0])
prikazi_subplot(ax, img_mirror, f'c) Zrcaljena slika (fliplr)\n{img_mirror.shape}')
print(f"c) Zrcaljene: np.fliplr(img) -> img[:, ::-1]")

# d) Smanjenje rezolucije (subsampling)
n = 10
img_small = img[::n, ::n]
ax = fig.add_subplot(gs[1, 1])
prikazi_subplot(ax, img_small, f'd) RES. x{n} MANJA\n{img.shape} -> {img_small.shape}')
print(f"d) Subsampling x{n}: {img.shape} -> {img_small.shape}")

# e) Samo druga cetvrtina po sirini
q1 = W // 4
q2 = W // 2
img_quarter = np.zeros_like(img)
img_quarter[:, q1:q2] = img[:, q1:q2]
ax = fig.add_subplot(gs[1, 2])
prikazi_subplot(ax, img_quarter, f'e) Samo 2. cetvrtina (stupci {q1}-{q2})\nostatak crn')
print(f"e) 2. cetvrtina: stupci {q1} - {q2} od {W}")

plt.show()

# ZADATAK 4 - GENERIRANJE SAHOVNICE
print("\n" + "="*60)
print("ZADATAK 4 - GENERIRANJE SAHOVNICE")
print("="*60)

def generiraj_sahovnicu(k, nv, ns):
    crni = np.zeros((k, k), dtype=np.uint8)
    bijeli = np.ones((k, k), dtype=np.uint8) * 255
    trake = []
    for i in range(nv):
        red = []
        for j in range(ns):
            red.append(crni if (i + j) % 2 == 0 else bijeli)
        trake.append(np.hstack(red))
    return np.vstack(trake)

sah = generiraj_sahovnicu(k=50, nv=4, ns=5)

print(f"Sahovnica: shape={sah.shape}, dtype={sah.dtype}")

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.imshow(sah, cmap='gray', vmin=0, vmax=255)
ax.set_title(f'k=50, nv=4, ns=5 → {sah.shape[1]}×{sah.shape[0]} px')
plt.suptitle('Zadatak 4 – Šahovnica (zeros, ones, hstack, vstack)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
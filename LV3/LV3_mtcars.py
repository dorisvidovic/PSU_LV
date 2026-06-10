import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mtcars = pd.read_csv('mtcars.csv')

print("=" * 65)
print("  ZADATAK 1 – ANALIZA PODATAKA (mtcars)")
print("=" * 65)
print(f"\nDataset: {len(mtcars)} automobila, {len(mtcars.columns)} stupaca")
print(f"Stupci : {list(mtcars.columns)}\n")

print("─" * 65)
print("1. Pet automobila s NAJVEĆOM potrošnjom (najmanji mpg):")
top5 = mtcars.sort_values('mpg', ascending=True).head(5)[['car', 'cyl', 'mpg']]
print(top5.to_string(index=False))

print("\n─" * 65)
print("2. Tri automobila s 8 cil. s NAJMANJOM potrošnjom (najveći mpg):")
top3_8cyl = (mtcars[mtcars.cyl == 8]
             .sort_values('mpg', ascending=False)
             .head(3)[['car', 'cyl', 'mpg']])
print(top3_8cyl.to_string(index=False))

print("\n─" * 65)
print("3. Srednja potrošnja za automobile s 6 cilindara:")
mean_6cyl = mtcars[mtcars.cyl == 6]['mpg'].mean()
print(f"   Srednja vrijednost: {mean_6cyl:.4f} mpg")

print("\n─" * 65)
print("4. Srednja potrošnja – 4 cil., masa 2000–2200 lbs (wt 2.0–2.2):")
filter_4 = mtcars[(mtcars.cyl == 4) &
                  (mtcars.wt >= 2.0) &
                  (mtcars.wt <= 2.2)]

print(filter_4[['car', 'cyl', 'wt', 'mpg']].to_string(index=False))

mean_4_wt = filter_4['mpg'].mean()
print(f"\n   Srednja vrijednost: {mean_4_wt:.4f} mpg")

print("\n─" * 65)
print("5. Broj automobila prema vrsti mjenjača:")
mjenjac = mtcars['am'].value_counts().sort_index()
print(f"   Automatski mjenjač (am=0) : {mjenjac[0]} automobila")
print(f"   Ručni mjenjač      (am=1) : {mjenjac[1]} automobila")

print("\n─" * 65)
print("6. Automatski mjenjač (am=0) i snaga > 100 hp:")
auto_100hp = mtcars[(mtcars.am == 0) & (mtcars.hp > 100)]
print(f"   Ukupno: {len(auto_100hp)} automobila")
print(auto_100hp[['car', 'am', 'hp']].to_string(index=False))

print("\n─" * 65)
print("7. Masa automobila u kilogramima:")
mtcars['wt_kg'] = (mtcars['wt'] * 1000 * 0.453592).round(2)
print(mtcars[['car', 'wt', 'wt_kg']].to_string(index=False))

print("\n" + "=" * 65)
print("  ZADATAK 2 – VIZUALIZACIJA")
print("=" * 65)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
mean_by_cyl = mtcars.groupby('cyl')['mpg'].mean()
bars = ax1.bar(
    ['4 cilindra', '6 cilindara', '8 cilindara'],
    mean_by_cyl.values,
    color=['#2196F3', '#4CAF50', '#FF5722']
)

for bar, val in zip(bars, mean_by_cyl.values):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.3,
        f'{val:.1f}',
        ha='center'
    )

ax1.set_title('Prosječna potrošnja po broju cilindara')
ax1.set_xlabel('Broj cilindara')
ax1.set_ylabel('Prosječni mpg')

ax2 = axes[0, 1]
data_wt = [mtcars[mtcars.cyl == c]['wt'].values for c in [4, 6, 8]]
ax2.boxplot(
    data_wt,
    labels=['4 cil.', '6 cil.', '8 cil.'],
    patch_artist=True
)
ax2.set_title('Distribucija mase po broju cilindara')
ax2.set_xlabel('Broj cilindara')
ax2.set_ylabel('Masa (1000 lbs)')

ax3 = axes[1, 0]
data_am = [
    mtcars[mtcars.am == 0]['mpg'].values,
    mtcars[mtcars.am == 1]['mpg'].values
]
ax3.boxplot(
    data_am,
    labels=['Automatski (am=0)', 'Ručni (am=1)'],
    patch_artist=True
)
ax3.set_title('Potrošnja: Automatski vs Ručni mjenjač')
ax3.set_xlabel('Vrsta mjenjača')
ax3.set_ylabel('mpg')

ax4 = axes[1, 1]

for am_val, marker in zip([0, 1], ['o', '^']):
    subset = mtcars[mtcars.am == am_val]
    ax4.scatter(
        subset['hp'],
        subset['mpg'],
        marker=marker,
        s=80
    )

ax4.set_title('Snaga (hp) vs Potrošnja (mpg) po mjenjaču')
ax4.set_xlabel('Snaga motora (hp)')
ax4.set_ylabel('Potrošnja (mpg)')

plt.tight_layout()
plt.show()

print("\nGotovo!")
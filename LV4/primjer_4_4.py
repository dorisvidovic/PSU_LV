import pandas as pd  # biblioteka za rad s tabličnim podacima (DataFrame)

# ========================
# UČITAVANJE PODATAKA
# ========================
df = pd.read_csv('cars_processed.csv')  # učitavanje CSV datoteke u pandas DataFrame
# df.head() -> možeš koristiti za pregled prvih 5 redaka
# df.info() -> pregled informacija o stupcima i tipu podataka

# ========================
# 1. BROJ MJERENJA
# ========================
print("1. Broj automobila u datasetu:")
print(len(df))  # broj redaka u DataFrameu = broj automobila
# alternativno: df.shape[0]

# ========================
# 2. TIPOVI STUPACA
# ========================
print("\n2. Tipovi stupaca:")
print(df.dtypes)  # tipovi podataka po stupcima (int, float, object)
# korisno za provjeru koji stupci su numerički, a koji kategorijski

# ========================
# 3. NAJSKUPLJI I NAJJEFTINIJI AUTOMOBIL
# ========================
max_price_row = df.loc[df['selling_price'].idxmax()]  # red s najvećom prodajnom cijenom
min_price_row = df.loc[df['selling_price'].idxmin()]  # red s najmanjom prodajnom cijenom

print("\n3. Najskuplji automobil:")
print(max_price_row[['name', 'selling_price']])  # ispis imena i cijene

print("\nNajjeftiniji automobil:")
print(min_price_row[['name', 'selling_price']])

# ========================
# 4. BROJ AUTOMOBILA PROIZVEDENIH 2012
# ========================
cars_2012 = df[df['year'] == 2012]  # filtriranje redaka gdje je godina proizvodnje 2012
print("\n4. Broj automobila proizvedenih 2012:")
print(len(cars_2012))
# alternativno: cars_2012.shape[0]

# ========================
# 5. NAJVIŠE I NAJMANJE KILOMETARA
# ========================
max_km_row = df.loc[df['km_driven'].idxmax()]  # red s najvećom kilometražom
min_km_row = df.loc[df['km_driven'].idxmin()]  # red s najmanjom kilometražom

print("\n5. Najviše kilometara:")
print(max_km_row[['name', 'km_driven']])

print("\nNajmanje kilometara:")
print(min_km_row[['name', 'km_driven']])

# ========================
# 6. NAJČEŠĆI BROJ SJEDALA
# ========================
most_common_seats = df['seats'].mode()[0]  # najčešći broj sjedala u datasetu
print("\n6. Najčešći broj sjedala:")
print(most_common_seats)
# mode() vraća niz svih modova, [0] uzima prvi ako ima više

# ========================
# 7. PROSJEČNA KILOMETRAŽA PO TIPU GORIVA
# ========================
avg_km_diesel = df[df['fuel'] == 'Diesel']['km_driven'].mean()  # prosjek za Diesel
avg_km_petrol = df[df['fuel'] == 'Petrol']['km_driven'].mean()  # prosjek za Petrol

print("\n7. Prosječna kilometraža:")
print("Diesel:", avg_km_diesel)
print("Petrol:", avg_km_petrol)
# može se proširiti na sve tipove goriva:
# df.groupby('fuel')['km_driven'].mean()
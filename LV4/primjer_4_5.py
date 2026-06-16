# ========================
# 1. UVOZ PAKETA
# ========================
import pandas as pd                                    # rad s tabličnim podacima (DataFrame)
from sklearn.model_selection import train_test_split   # podjela podataka na treniranje i test
from sklearn.preprocessing import StandardScaler      # skaliranje značajki
from sklearn.linear_model import LinearRegression     # linearna regresija
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error  
# različite metrike za evaluaciju modela

# ========================
# 2. UČITAVANJE PODATAKA
# ========================
df = pd.read_csv('cars_processed.csv')  
# učitavanje CSV datoteke u pandas DataFrame
# df.shape -> provjerava dimenzije
# df.head() -> prikaz prvih 5 redaka radi vizualne kontrole

# ========================
# 3. UKLANJANJE NEPOTREBNIH STUPACA
# ========================
df = df.drop(['name'], axis=1)
# 'name' je tekstualni identifikator koji ne pomaže modelu
# axis=1 znači da uklanjamo stupac (axis=0 bi uklonio red)

# ========================
# 4. ONE-HOT ENCODING KATEGORIČKIH VARIJABLI
# ========================
df = pd.get_dummies(df, drop_first=True)
# pretvaramo kategoričke varijable u numeričke stupce 0/1
# drop_first=True -> izbjegavanje dummy variable trap (multikolinearnosti)

# ========================
# 5. DEFINICIJA ULIZA (X) I IZLAZA (y)
# ========================
X = df.drop('selling_price', axis=1)  # sve osim ciljne varijable
y = df['selling_price']               # ciljna varijabla (cijena automobila)

# ========================
# 6. PODJELA NA TRENING I TEST SKUP
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 20% podataka za test, 80% za treniranje
# random_state=42 -> reproducibilnost permutacije
# Može se mijenjati test_size npr. 0.3 ili 0.25, ili random_state za različite permutacije

# ========================
# 7. SKALIRANJE ZNAČAJKI
# ========================
scaler = StandardScaler()                 # standardizacija (0 srednja vrijednost, std=1)
X_train_scaled = scaler.fit_transform(X_train)  # fit + transformiranje trening skupa
X_test_scaled = scaler.transform(X_test)       # transformiranje test skupa po parametrima treninga
# StandardScaler je bitan kada značajke imaju različite jedinice ili raspon
# Alternativa: MinMaxScaler, RobustScaler

# ========================
# 8. TRENING LINEARNOG MODELA
# ========================
model = LinearRegression()           # kreiranje instance modela
model.fit(X_train_scaled, y_train)   # učenje koeficijenata

# ========================
# 9. PREDIKCIJA
# ========================
y_train_pred = model.predict(X_train_scaled)  # predikcija na trening podacima
y_test_pred = model.predict(X_test_scaled)    # predikcija na test podacima

# ========================
# 10. EVALUACIJA MODELA
# ========================
print("=== REZULTATI MODELA ===")

print("\n--- TRAIN ---")
print("MAE:", mean_absolute_error(y_train, y_train_pred))  # srednja apsolutna pogreška
print("MSE:", mean_squared_error(y_train, y_train_pred))   # srednja kvadratna pogreška
print("R2:", r2_score(y_train, y_train_pred))              # koeficijent determinacije (koliko varijacije objašnjava model)
print("Max error:", max_error(y_train, y_train_pred))      # najveća pojedinačna pogreška

print("\n--- TEST ---")
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("R2:", r2_score(y_test, y_test_pred))
print("Max error:", max_error(y_test, y_test_pred))

# ========================
# KLJUČNI PARAMETRI I OPCIJE KOJE SE MOGU PROMIJENITI
# ========================
# 1. test_size u train_test_split -> omjer podjele podataka (npr. 0.3 za 70/30)
# 2. random_state -> reproducibilnost ili različite permutacije podataka
# 3. StandardScaler -> može se zamijeniti MinMaxScaler ili RobustScaler
# 4. LinearRegression -> može se zamijeniti Ridge, Lasso, ElasticNet radi regularizacije
# 5. Metrike -> možeš dodati npr. median_absolute_error, explained_variance_score
# 6. One-hot encoding -> drop_first=True može biti False, ili koristiti sklearn OneHotEncoder
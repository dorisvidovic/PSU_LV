# ========================
# UVOZ PAKETA
# ========================
import numpy as np                          # rad s numeričkim poljima i matematičkim funkcijama
import matplotlib.pyplot as plt             # crtanje grafova i vizualizacija podataka
from sklearn.linear_model import LinearRegression  # linearna regresija
from sklearn.metrics import mean_squared_error    # izračun srednje kvadratne pogreške
from sklearn.preprocessing import PolynomialFeatures  # generiranje polinomskih značajki

# ========================
# DEFINICIJA NELINEARNE FUNKCIJE
# ========================
def non_func(x):
    """
    Stvarna funkcija s kombinacijom sinusnih i kosinusnih valova.
    Parametar:
        x : np.array (1D)
    Povratna vrijednost:
        y : np.array (1D)
    """
    return 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) \
           - 1.1622*np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)

# ========================
# DODAVANJE ŠUMA
# ========================
def add_noise(y):
    """
    Dodaje Gaussov šum u podatke proporcionalno rasponu y.
    Parametar:
        y : np.array
    Povratna vrijednost:
        y_noisy : np.array
    """
    np.random.seed(14)  # reproducibilnost šuma
    varNoise = np.max(y) - np.min(y)  # skaliranje šuma prema rasponu podataka
    return y + 0.1*varNoise*np.random.normal(0,1,len(y))
    # 0.1 -> intenzitet šuma, može se promijeniti (npr. 0.2 ili 0.05)

# ========================
# POSTAVLJANJE PARAMETARA
# ========================
degrees = [5, 10, 15]  # stupnjevi polinoma koje ćemo testirati
N = 250                # broj uzoraka (može biti 20, 50, 200 itd.)

# ========================
# GENERIRANJE PODATAKA
# ========================
x = np.linspace(1, 10, N)       # ravnomjerno raspoređene točke između 1 i 10
y_true = non_func(x)             # stvarna funkcija
y_measured = add_noise(y_true)   # mjerenja s dodanim šumom

# sklearn LinearRegression zahtijeva 2D array
x = x[:, np.newaxis]             # pretvaranje u stupčasti vektor
y_measured = y_measured[:, np.newaxis]

# ========================
# PODJELA PODATAKA NA TRENING I TEST
# ========================
np.random.seed(12)                          # reproducibilnost permutacije
indeksi = np.random.permutation(len(x))    # nasumična permutacija indeksa
split = int(0.7*len(x))                     # 70% za treniranje, 30% za test

indeksi_train = indeksi[:split]            # indeksi trening skupa
indeksi_test = indeksi[split:]             # indeksi test skupa

# ========================
# SPREMANJE MSE
# ========================
MSEtrain = []   # spremnik za MSE trening
MSEtest = []    # spremnik za MSE test

# ========================
# PLOT STVARNE FUNKCIJE
# ========================
plt.figure(figsize=(10,6))
plt.plot(x, y_true, 'k', label='stvarna funkcija')

# ========================
# PETLJA KROZ RAZNE STUPNJE POLINOMA
# ========================
for deg in degrees:
    
    # 1. Generiranje polinomskih značajki (x, x^2, x^3,...x^deg)
    poly = PolynomialFeatures(degree=deg)
    x_poly = poly.fit_transform(x)  # x_poly.shape = (N, deg+1)
    
    # 2. Podjela na train/test
    xtrain = x_poly[indeksi_train]
    ytrain = y_measured[indeksi_train]
    
    xtest = x_poly[indeksi_test]
    ytest = y_measured[indeksi_test]
    
    # 3. Treniranje linearnog modela
    model = LinearRegression()
    model.fit(xtrain, ytrain)  # učenje koeficijenata polinoma
    
    # 4. Predikcija
    ytrain_p = model.predict(xtrain)  # predikcija na trening podacima
    ytest_p = model.predict(xtest)    # predikcija na test podacima
    
    # 5. MSE
    MSEtrain.append(mean_squared_error(ytrain, ytrain_p))  # srednja kvadratna pogreška trening
    MSEtest.append(mean_squared_error(ytest, ytest_p))     # srednja kvadratna pogreška test
    
    # 6. Predikcija na cijelom x za crtanje glatke krivulje
    y_model = model.predict(x_poly)
    
    # 7. Plot modela
    plt.plot(x, y_model, label=f'degree={deg}')

# ========================
# PLOT TRENING TOČAKA
# ========================
plt.scatter(x[indeksi_train], y_measured[indeksi_train], c='black', label='train', s=20)

# Finalni graf
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Usporedba modela za različite stupnjeve polinoma')
plt.show()

# ========================
# ISPIS MSE
# ========================
print("MSEtrain:", MSEtrain)
print("MSEtest:", MSEtest)

# ========================
# KLJUČNI PARAMETRI KOJI SE MOGU PROMIJENITI
# ========================
# 1. degrees = [5,10,15] -> više ili manje fleksibilnosti modela
# 2. N = 250 -> broj uzoraka (manje/mnogo)
# 3. omjer train/test (trenutno 70/30) -> npr. 80/20
# 4. intenzitet šuma u add_noise -> npr. 0.05, 0.2
# 5. LinearRegression -> može se zamijeniti Ridge/Lasso radi regularizacije
# 6. seed-ovi -> reproducibilnost ili randomizacija
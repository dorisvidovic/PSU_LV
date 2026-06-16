# Uvoz potrebnih biblioteka
import numpy as np                       # NumPy: rad s numeričkim poljima, vektorima, matricama, matematičke funkcije
import matplotlib.pyplot as plt          # Matplotlib: crtanje grafova i vizualizacija podataka
import sklearn.linear_model as lm        # Sklearn: linearni modeli (regresija)
from sklearn.metrics import mean_squared_error  # Funkcija za izračun srednje kvadratne pogreške
from sklearn.preprocessing import PolynomialFeatures  # Pretvaranje ulaznih podataka u polinomske značajke

# ========================
# DEFINICIJA NELINEARNE FUNKCIJE
# ========================
def non_func(x):
    """
    Funkcija koja generira "stvarne" podatke.
    Kombinira sin i cos funkcije različitih frekvencija.
    
    x : numpy array
        Ulazni podaci (1D niz)
    
    returns : numpy array
        Izlazni podaci
    """
    y = (
        1.6345
        - 0.6235 * np.cos(0.6067 * x)   # prva harmonika cos
        - 1.3501 * np.sin(0.6067 * x)   # prva harmonika sin
        - 1.1622 * np.cos(2 * x * 0.6067)  # druga harmonika cos
        - 0.9443 * np.sin(2 * x * 0.6067)  # druga harmonika sin
    )
    return y  # vraća vektor y istih dimenzija kao x

# ========================
# DODAVANJE ŠUMA U PODATKE
# ========================
def add_noise(y):
    """
    Funkcija dodaje Gaussov šum u podatke.
    
    y : numpy array
        Ulazni podaci bez šuma
    
    returns : numpy array
        Ulazni podaci sa šumom
    """
    np.random.seed(14)  # postavljanje seed-a za reproducibilnost slučajnog šuma
    
    varNoise = np.max(y) - np.min(y)  # skaliranje šuma prema rasponu podataka
    y_noisy = y + 0.1 * varNoise * np.random.normal(0, 1, len(y))
    # 0.1 -> intenzitet šuma (može se mijenjati za više/manje šuma)
    # np.random.normal(0,1,len(y)) -> generira normalno distribuirane brojeve sa srednjom 0 i std 1
    
    return y_noisy

# ========================
# GENERIRANJE PODATAKA
# ========================
x = np.linspace(1, 10, 200)  # 200 točaka od 1 do 10
y_true = non_func(x)          # stvarna funkcija
y_measured = add_noise(y_true)  # dodavanje šuma

# sklearn LinearRegression očekuje 2D polje (n_samples, n_features)
x = x[:, np.newaxis]           # pretvaranje 1D u 2D (stupac)
y_measured = y_measured[:, np.newaxis]

# ========================
# POLINOMSKA TRANSFORMACIJA
# ========================
poly = PolynomialFeatures(degree=10)  # generiranje polinomskih značajki do 10. stupnja
xnew = poly.fit_transform(x)
# xnew će sada imati stupce: [1, x, x^2, x^3, ..., x^10]
# - 1 je bias (intercept) za linearnu regresiju
# - zamjena degree omogućava više ili manje fleksibilnosti modela
# - veći stupanj = model bolje prati šum i nelinearnost, ali postoji overfitting rizik

# ========================
# PODJELA NA TRENING I TEST SKUP
# ========================
np.random.seed(12)  # reproducibilnost permutacije
indeksi = np.random.permutation(len(xnew))  # nasumični redoslijed indeksa

# 70% za treniranje
indeksi_train = indeksi[0:int(np.floor(0.7 * len(xnew)))]

# ostatak za testiranje
indeksi_test = indeksi[int(np.floor(0.7 * len(xnew)))+1 : len(xnew)]

# formiranje train i test podataka
xtrain = xnew[indeksi_train, ]
ytrain = y_measured[indeksi_train]

xtest = xnew[indeksi_test, ]
ytest = y_measured[indeksi_test]

# ========================
# TRENING LINEARNOG MODELA
# ========================
linearModel = lm.LinearRegression()  # instanca linearne regresije
linearModel.fit(xtrain, ytrain)      # učenje koeficijenata na polinomskim značajkama

# ========================
# PREDIKCIJA I EVALUACIJA
# ========================
ytest_p = linearModel.predict(xtest)  # predikcija na test skupu
MSE_test = mean_squared_error(ytest, ytest_p)  # srednja kvadratna pogreška
print("MSE test:", MSE_test)

# ========================
# PLOT PREDIKCIJA NA TESTU
# ========================
plt.figure(1)
plt.plot(xtest[:, 1], ytest_p, 'og', label='predicted')  # zelene točke = predikcija
plt.plot(xtest[:, 1], ytest, 'or', label='test')         # crvene točke = stvarni test
plt.legend(loc=4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predikcija vs Test podaci')
plt.show()

# ========================
# PLOT STVARNOG SIGNALA VS MODEL
# ========================
plt.figure(2)
plt.plot(x, y_true, label='stvarna funkcija f(x)')          # stvarna funkcija
plt.plot(x, linearModel.predict(xnew), 'r-', label='model') # model na cijelom x (glatka linija)
plt.plot(xtrain[:, 1], ytrain, 'ok', label='train')         # train točke
plt.xlabel('x')
plt.ylabel('y')
plt.title('Model vs Stvarna funkcija')
plt.legend(loc=4)
plt.show()

# ========================
# KLJUČNE STVARI KOJE SE MOGU KONFIGURIRATI
# ========================
# 1. degree u PolynomialFeatures -> fleksibilnost modela
# 2. intenzitet šuma u add_noise (0.1*varNoise) -> više/manje šuma
# 3. veličina train/test podjele (trenutno 70/30) -> može biti 80/20 ili k-fold
# 4. LinearRegression može se zamijeniti Ridge/Lasso za regularizaciju
# 5. Broj točaka x (trenutno 200) -> gustoća mreže, bolja vizualizacija
# 6. seed za reproducibilnost -> može se promijeniti ili ukloniti za randomizaciju
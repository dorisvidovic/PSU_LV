# Uvoz potrebnih biblioteka
import numpy as np                      # za numeričke operacije i rad s poljima (array)
import matplotlib.pyplot as plt         # za crtanje grafova
import sklearn.linear_model as lm       # za linearne modele (regresija)
from sklearn.metrics import mean_squared_error  # za izračun pogreške modela


# Funkcija koja generira "stvarne" podatke (neka složena nelinearna funkcija)
def non_func(x):
    # Kombinacija sinusnih i kosinusnih funkcija (periodični signal)
    y = (
        1.6345
        - 0.6235 * np.cos(0.6067 * x)
        - 1.3501 * np.sin(0.6067 * x)
        - 1.1622 * np.cos(2 * x * 0.6067)
        - 0.9443 * np.sin(2 * x * 0.6067)
    )
    return y


# Funkcija koja dodaje šum (noise) u podatke
def add_noise(y):
    np.random.seed(14)  # postavljamo seed da dobijemo uvijek isti "random" rezultat

    # Varijanca šuma ovisi o rasponu podataka
    varNoise = np.max(y) - np.min(y)

    # Dodavanje Gaussovog šuma (normalna distribucija)
    y_noisy = y + 0.2 * varNoise * np.random.normal(0, 1, len(y))

    return y_noisy


# Generiranje x vrijednosti (100 točaka između 1 i 10)
x = np.linspace(1, 10, 100)

# Stvarne vrijednosti bez šuma
y_true = non_func(x)

# Simulirane izmjerene vrijednosti (sa šumom)
y_measured = add_noise(y_true)


# =======================
# CRTANJE STVARNIH I MJERENIH PODATAKA
# =======================
plt.figure(1)
plt.plot(x, y_measured, 'ok', label='mjereno')  # crne točke (mjerenja)
plt.plot(x, y_true, label='stvarno')            # stvarna funkcija
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.show()


# =======================
# PODJELA NA TRAIN I TEST SKUP
# =======================

np.random.seed(12)  # seed za reproducibilnost

# Nasumična permutacija indeksa
indeksi = np.random.permutation(len(x))

# 70% podataka za treniranje
indeksi_train = indeksi[0:int(np.floor(0.7 * len(x)))]

# ostatak za testiranje
indeksi_test = indeksi[int(np.floor(0.7 * len(x))) + 1:len(x)]

# Pretvaranje u stupčaste vektore (sklearn očekuje 2D ulaz)
x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]

# Formiranje train i test skupova
xtrain = x[indeksi_train]
ytrain = y_measured[indeksi_train]

xtest = x[indeksi_test]
ytest = y_measured[indeksi_test]


# Vizualizacija train/test podjele
plt.figure(2)
plt.plot(xtrain, ytrain, 'ob', label='train')  # plavo = train
plt.plot(xtest, ytest, 'or', label='test')     # crveno = test
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.show()


# =======================
# LINEARNA REGRESIJA
# =======================

# Kreiranje modela
linearModel = lm.LinearRegression()

# Učenje modela (fitanje na train podacima)
linearModel.fit(xtrain, ytrain)

# Ispis modela (y = Theta0 + Theta1 * x)
print('Model je oblika y_hat = Theta0 + Theta1 * x')
print('y_hat = ', linearModel.intercept_, '+', linearModel.coef_, '* x')


# =======================
# TESTIRANJE MODELA
# =======================

# Predikcija na test skupu
ytest_p = linearModel.predict(xtest)

# Izračun srednje kvadratne pogreške (MSE)
MSE_test = mean_squared_error(ytest, ytest_p)
print("MSE na test skupu:", MSE_test)


# =======================
# CRTANJE REZULTATA
# =======================

plt.figure(3)

# Predviđene vrijednosti (zeleno)
plt.plot(xtest, ytest_p, 'og', label='predicted')

# Stvarne test vrijednosti (crveno)
plt.plot(xtest, ytest, 'or', label='test')

plt.legend(loc=4)

# Crtanje regresijskog pravca
x_pravac = np.array([1, 10])[:, np.newaxis]  # dvije točke za liniju
y_pravac = linearModel.predict(x_pravac)

plt.plot(x_pravac, y_pravac, label='linearni model')
plt.show()
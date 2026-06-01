# ZADATAK 1
# traži od korisnika broj radnih sati i zaradu po satu, 
# a zatim izračunava ukupnu zaradu.
# Nakon toga, koristi funkciju za izračunavanje ukupne zarade i ispisuje rezultat.

print("=" * 50)
print("ZADATAK 1 - Radni sati i zarada")
print("=" * 50)

# bez funckije

# input() uvijek vraca string, pa ga moramo pretvoriti u float()
sati = float(input("Radni sati: "))
eura_po_satu = float(input("eura/h: "))

# mnozenje sati i eura po satu daje ukupnu zaradu
ukupno = sati * eura_po_satu

# f-string omogucava umetanje varijable direktno u tekst
print(f"Ukupno: {ukupno:.2f} eura")


# sa funkcijom
def total_euro(sati, eura_po_satu):
    return sati * eura_po_satu # return vraca rezultat funkcije, a ne ispisuje ga

# pozivanje fje i spremanje rezultata u varijablu
zarada = total_euro(sati, eura_po_satu)
print(f"(Funkcija) Ukupno: {zarada:.2f} eura")


# ZADATAK 2
# program traži unos ocjene između 0.0 i 1.0
# zatim ispisuje slovnu ocjenu (A-F)
# ako je unos kriv, traži ponovni unos

print()
print("=" * 50)
print("ZADATAK 2 - Ocjene")
print("=" * 50)

while True:
    try:
        # pretvaramo unos u float
        ocjena = float(input("Unesite ocjenu (0.0 - 1.0): "))

        # provjeravamo je li ocjena u ispravnom intervalu
        if ocjena < 0.0 or ocjena > 1.0:
            print("Pogreska: broj mora biti između 0.0 i 1.0!")
            continue # nastavlja petlju od pocetka, bez izvrsavanja koda ispod

        # odredivanje ocjene
        if ocjena >= 0.9:
            print("Ocjena: A")  
        elif ocjena >= 0.8:
            print("Ocjena: B")
        elif ocjena >= 0.7:
            print("Ocjena: C")
        elif ocjena >= 0.6:
            print("Ocjena: D")
        else:
            print("Ocjena: F")

        break

    except ValueError:
        # ValueError se dogodi kada float() ne može pretvoriti unos
        print("Pogreska: niste unijeli broj!")


# ZADATAK 3
# program u beskonacnoj petlji trazi unos brojeva od korisnika
# korisnik moze unijeti "Done" da prekine unos
# na kraju se ispisuju statistike i sortirana lista unesenih brojeva

print()
print("=" * 50)
print("ZADATAK 3 - Unos brojeva u petlji")
print("=" * 50)

brojevi = []     # prazna lista u koju cemo spremati unesene brojeve

while True:     # beskonacna petlja, nastavlja se dok korisnik ne unese "Done"
    unos = input("Unesite broj (ili 'Done' za kraj): ")

    if unos.strip().lower() == "done":      #uvjet za prekid petlje, ignorira razmake i velika/mala slova
        break

    try:
        broj = float(unos)      # pokusavamo pretvoriti unos u broj, ako nije broj, baca se ValueError
        brojevi.append(broj)    # .append() dodaje broj na kraj liste brojevi

    except ValueError:
        # ako unos nije broj, ispisujemo poruku o pogresci i nastavljamo petlju
        print("Pogreska: niste unijeli broj!")

# provjeravamo je li korisnik unio barem jedan broj prije nego sto pokusamo izracunati statistike
if len(brojevi) == 0:
    print("Niste unijeli nijedan broj.")

else:
    print(f"\nBroj unesenih vrijednosti: {len(brojevi)}")
    print(f"Srednja vrijednost: {sum(brojevi) / len(brojevi):.2f}")
    print(f"Minimalna vrijednost: {min(brojevi)}")
    print(f"Maksimalna vrijednost: {max(brojevi)}")

    # sorted() funkcija vraca novu listu koja je sortirana, originalna lista brojevi ostaje nepromijenjena
    print(f"Sortirana lista: {sorted(brojevi)}")


# ZADATAK 4
# program cita tekstualnu datoteku red po red
# trazi linije koje pocinju sa "X-DSPAM-Confidence:"
# iz tih linija izvlači broj i racuna prosjecnu vrijednost

print()
print("=" * 50)
print("ZADATAK 4 - DSPAM pouzdanost")
print("=" * 50)

ime_datoteke = input("Unesite ime datoteke: ")

try:
    # open() otvara datoteku, a encoding="utf-8" osigurava da se datoteka ispravno cita bez obzira na znakove koje sadrzi
    fhand = open(ime_datoteke, encoding="utf-8")

    ukupno_pouzdanosti = 0.0        # suma svih pronadenih vrijednosti
    broj_linija = 0                 # broj pronadenih linija

    for linija in fhand:            # citamo datoteku red po red
        linija = linija.strip()     # .rstrip() uklanja razmake i znakove za novi red sa pocetka i kraja linije

        # .startswith() provjerava pocinje li linija sa zadanim tekstom
        if linija.startswith("X-DSPAM-Confidence:"):
            broj_linija += 1

            # .split() dijeli string po razmacima u listu dijelova
            dijelovi = linija.split()

            # drugi element (indeks 1) je broj koji nas zanima
            vrijednost = float(dijelovi[1])
            ukupno_pouzdanosti += vrijednost

    fhand.close() # uvijek zatvoriti datoteku nakon citanja

    if broj_linija == 0:
        print("Nisu pronadene X-DSPAM-Confidence linije u datoteci.")
    else: 
        prosjek_pouzdanosti = ukupno_pouzdanosti / broj_linija
        print(f"Average X-DSPAM-Confidence: {prosjek_pouzdanosti}")

except FileNotFoundError:
    # ukoliko datoteka s tim imenom ne postoji
    print(f"Pogreska: datoteka '{ime_datoteke}' nije pronadena!")


# ZADATAK 5
# citamo datoteku song.txt i brojimo koliko se puta svaka rijec pojavljuje
# koristimo rjecnik (dictionary)

print()
print("=" * 50)
print("ZADATAK 5 - Frekvencija rijeci u song.txt")
print("=" * 50)

rjecnik_rijeci = {} # prazan rjecnik, kljucevi ce biti rijeci, a vrijednosti broj pojavljivanja

try:
    fhand = open("song.txt", encoding="utf-8")

    for linija in fhand: 
        linija = linija.rstrip()

        # .lower() pretvara sva slova u mala slova
        # .split() dijeli liniju u listu pojedinacnih rijeci
        rijeci = linija.lower().split()

        for rijec in rijeci:
            # ako rijec vec postoji u rjecniku, povecamo brojac
            # ako ne postoji, .get() vraca 0 (zadani default) i onda dodajemo 1
            rjecnik_rijeci[rijec] = rjecnik_rijeci.get(rijec, 0) + 1

    fhand.close()

    # pronalazimo rijeci koje se pojavljuju samo jednom
    # List comprehension: kratak zapis petlje koja gradi listu
    jednom = [rijec for rijec, broj in rjecnik_rijeci.items() if broj == 1]

    print(f"Ukupno razlicitih rijeci: {len(rjecnik_rijeci)}")
    print(f"Rijeci koje se pojavljuju samo jednom ({len(jednom)} kom.): ")
    for r in sorted(jednom): # sorted() ispisuje abecednim redom
        print(f"  {r}")

except FileNotFoundError:
    print("Pogreska: datoteka 'song.txt' nije pronadena!")


# ZADATAK 6
# Čitamo SMSSpamCollection.txt.
# Svaka linija počinje s "ham" ili "spam", zatim tabulator (\t),
# zatim tekst poruke.
# a) Prosječan broj riječi za ham i spam poruke
# b) Koliko spam poruka završava uskličnikom

print()
print("=" * 50)
print("ZADATAK 6 - SMS Spam analiza ")
print("=" * 50)

# dvije liste u koje cemo spremati broj rijeci za ham i spam poruke
ham_broj_rijeci = []
spam_broj_rijeci = []

spam_s_usklicnikom = 0      # brojac spam poruka koje zavrsavaju s "!"

try:
    fhand = open("SMSSpamCollection.txt", encoding="utf-8")

    for linija in fhand:
        linija = linija.rstrip()

        # .split("\t", 1) dijeli liniju po tabulatoru, max 1 puta
        # → dobivamo ['ham', 'tekst poruke'] ili ['spam', 'tekst poruke']
        dijelovi = linija.split("\t", 1)

        # preskačemo linije koje nemaju točno 2 dijela (prazne linije itd.)
        if len(dijelovi) != 2:
            continue

        oznaka = dijelovi[0]
        poruka = dijelovi[1]

        # .split() dijeli poruku po razmacima i broji dobivene dijelove
        broj_rijeci = len(poruka.split())

        if oznaka == "ham":
            ham_broj_rijeci.append(broj_rijeci)
        elif oznaka == "spam":
            spam_broj_rijeci.append(broj_rijeci)

             # .rstrip() briše razmake s kraja, .endswith() provjerava zadnji zna
            if poruka.rstrip().endswith("!"):
                spam_s_usklicnikom += 1

    fhand.close()

    # --- Ispis rezultata ---
 
    # a) Prosječan broj riječi
    if ham_broj_rijeci:
        ham_prosjek = sum(ham_broj_rijeci) / len(ham_broj_rijeci)
        print(f"a) Prosjecan broj rijeci u HAM porukama: {ham_prosjek:.2f}")

    if spam_broj_rijeci:
        spam_prosjek = sum(spam_broj_rijeci) / len(spam_broj_rijeci)
        print(f"      Prosjecan broj rijeci u SPAM porukama: {spam_prosjek:.2f}")
   
    # b) Spam poruke s uskličnikom
        print(f"b) Broj SPAM poruka koje završavaju s '!': {spam_s_usklicnikom}")

except FileNotFoundError:
    print("Pogreska: datoteka 'SMSSpamCollection.txt' nije pronadena!")
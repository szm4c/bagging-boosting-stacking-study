# Autorzy Projektu
- Gabriela Jeznach
- Monika Jarosińska
- Karina Kownacka
- Mariusz Wawrzycki
- Szymon Pawłowski

# Przegląd Projektu

To repozytorium zawiera porównanie trzech popularnych metod ensemble-based dla problemu regresji:
- **Bagging (Random Forest)**
- **Boosting (XGBoost)**
- **Stacking (StackingRegressor z meta-learnerem Ridge)**

Analiza obejmuje sześć zróżnicowanych zbiorów danych:
- **Syntetyczne**: `regression`, `friedman1`, `friedman3`
- **Rzeczywiste**: `california_housing`, `energy_efficiency`, `airfoil_self_noise`

---

## Eksploracyjna Analiza Danych (EDA)

- Każdy zbiór danych jest wczytywany do pipeline’u wstępnego przetwarzania.  
- Przeprowadzamy podstawowe testy rozkładu (histogramy, wykresy pudełkowe), oceniamy korelacje cech (mapy cieplne, wykresy rozrzutu) oraz identyfikujemy wartości odstające.  
- Wszelkie niezbędne transformacje (np. skalowanie, kodowanie zmiennych kategorycznych) są stosowane w sposób spójny dla wszystkich zbiorów danych.

---

## Tuning Hiperparametrów

### Random Forest & XGBoost

- Wykorzystujemy Optuna do poszukiwania najlepszej kombinacji hiperparametrów na 90 % zbioru treningowego.  
- Zagnieżdżona walidacja krzyżowa (K-Fold CV) służy do wyboru optymalnych:
  - głębokości drzew (`max_depth`),
  - współczynników uczenia (`learning_rate`),
  - liczby estymatorów (`n_estimators`),
  - oraz innych parametrów specyficznych dla danego algorytmu.  

### Stacking (Ridge Meta-Learner)

- Meta-learner Ridge jest tuningowany za pomocą `RidgeCV` na podstawie tego samego podziału na foldy w zbiorze treningowym.  
- Wyszukiwany jest najlepszy parametr regularyzacji (`alpha`) dla regresora Ridge.

Po znalezieniu optymalnych hiperparametrów dla każdego bazowego modelu i każdego zbioru danych, „najlepsze” pipeline’y są ponownie trenowane na pełnym 90 % podziale i zapisywane do późniejszej ewaluacji.  

---

# Porównanie modeli

- **Procedura ewaluacji**  
  - **K-Fold Cross-Validation na zbiorze treningowym**  
    Każdy model trenowany jest na 90 % danych (z tym samym seedem). Dla każdego modelu i każdej metryki (RMSE, MAE, R²) zebrano wyniki z poszczególnych foldów i obliczono średnią oraz odchylenie standardowe.  
  - **Ocena na zewnętrznym zbiorze testowym (10 %)**  
    Po finalnym strojeniu hiperparametrów modele zostały wytrenowane na pełnych 90 % danych treningowych, a następnie ocenione na nieznanych wcześniej 10 %. Na tych danych obliczono ostateczne wartości RMSE, MAE i R², co daje bezstronną ocenę generalizacji.  

- **Metryki porównawcze**  
  - **RMSE** (Root Mean Squared Error): średnia pierwiastkowana z kwadratów różnic między przewidywaniami a wartościami rzeczywistymi, w jednostkach oryginalnego celu.  
  - **MAE** (Mean Absolute Error): średnia bezwzględna różnica między przewidywaniami a wartościami rzeczywistymi.  
  - **R²**: współczynnik determinacji, określający, jaka część wariancji zmiennej celu została wyjaśniona przez model.  

- **Tabela wyników**  
  - Dla każdego modelu i zestawu danych przedstawiono:  
    - Średni ± odchylenie standardowe RMSE, MAE i R² z CV.  
    - Odpowiednie wartości RMSE, MAE i R² na zbiorze testowym.  
    - Różnicę (gap) między średnią z CV a wynikiem testowym (overfitting gap).  
  - Wyniki prezentowane są w sformatowanym DataFrame z wyróżnieniem najlepszego rezultatu w każdej kolumnie.  

- **Dodatkowe analizy**  
  - **Rozkłady residuów**  
    - Histogramy błędów (predykcja – wartość rzeczywista) dla każdego modelu na zbiorze testowym, umożliwiające ocenę przesunięcia (bias) i rozrzutu reszt.   
  - **Ranking ważności cech**  
    - Dla RF i XGB wyznaczono `feature_importances_` i uszeregowano cechy według ich wkładu w prognozy.  
    - Stworzono skonsolidowany DataFrame z pozycją (Rank) dla każdej cechy w każdym modelu i zestawie danych.  

# Podsumowanie i Wnioski

### Kluczowe wyniki w pigułce

| Zbiór danych                       | Najlepszy wynik na zbiorze testowym*      |
|------------------------------------|-------------------------------------------|
| regression (1 k, syntetyczny)      | XGB (najniższy RMSE/MAE)                  |
| friedman1 (5 k, syntetyczny)       | **XGB**                                   |
| friedman3 (200, syntetyczny)       | **XGB**                                   |
| california housing (20 k, rzeczywisty)      | **XGB**                                   |
| airfoil self-noise (1.5 k, rzeczywisty)     | **XGB**                                   |
| energy efficiency (768, rzeczywisty) | **RF**                                    |

\*Zwycięzca = model z najniższym RMSE na zbiorze testowym (w razie remisu rozstrzyga MAE).

---

### Wnioski dla poszczególnych zbiorów danych

- **regression:** XGB uzyskuje najmniejsze błędy; Stacking ma nieznaczną przewagę w R².  
- **friedman1:** Na dużym, wysoko wymiarowym zbiorze syntetycznym XGB jest około 5 % lepszy od kolejnego modelu.  
- **friedman3:** Przy tylko 200 próbkach XGB wciąż wiedzie prym, choć różnice są niewielkie.  
- **california housing:** XGB wygrywa wyraźnie (> 10 % przewagi w RMSE).  
- **airfoil self-noise:** XGB wyprzedza RF i Stack o około 3 % w RMSE.  
- **energy efficiency:** RF zajmuje pierwsze miejsce z przewagą około 2 % w RMSE; XGB wykazuje lekkie przeuczenie.

---

### Ogólne spostrzeżenia

- **Boosting (XGB)** to najczęściej najlepszy wybór, zwłaszcza w przypadku większych lub bardziej złożonych danych.  
- **Bagging (RF)** to szybka i solidna baza, która może wygrywać na uporządkowanych, średniej wielkości zbiorach.  
- **Stacking** rzadko wygrywa z XGB; jego główną zaletą jest nieznaczny wzrost R² w kilku przypadkach, kosztem wyższego czasu obliczeń i złożoności.  
---

# Instrukcja korzystania z repozytorium

## 1. Struktura repozytorium

W repozytorium znajdziesz kilka głównych katalogów:
- src/ - tutaj znajdują się wszystkie pliki z kodem źródłowym.
- notebooks/ - tutaj zapisujemy nasze notebooki eksperymentalne.
- data/ - dane potrzebne do pracy.
- tests/ - tym się nie przejmujcie :)
- requirements.txt - lista wszystkich bibliotek potrzebnych do uruchomienia projektu.

## 2. Instalacja Git i klonowanie repozytorium lokalnie

### Instalacja Git
1. Przejdź na stronę [oficjalnej strony Git](https://git-scm.com/).
2. Pobierz instalator odpowiedni dla Twojego systemu operacyjnego (Windows, macOS, Linux).
3. Uruchom instalator i postępuj zgodnie z instrukcjami na ekranie. Zalecamy pozostawienie domyślnych ustawień.
4. Po instalacji sprawdź, czy Git działa poprawnie, wpisując w terminalu:
   ```
   git --version
   ```
   Powinno wyświetlić się coś w stylu `git version X.X.X`.

### Klonowanie repozytorium
1. Otwórz terminal lub PowerShell.
2. Przejdź do katalogu, w którym chcesz sklonować repozytorium, np.:
   ```
   cd C:\Users\TwojaNazwaUżytkownika\Documents
   ```
3. Wykonaj polecenie klonowania:
   ```
   git clone https://github.com/szm4c/bagging-boosting-stacking-study.git
   ```
4. Po zakończeniu klonowania przejdź do katalogu repozytorium:
   ```
   cd bagging-boosting-stacking-study
   ```

## 3. Instalacja Python i przygotowanie środowiska

### Instalacja Python
1. Przejdź na stronę [oficjalnej strony Python](https://www.python.org/).
2. Pobierz instalator odpowiedni dla Twojego systemu operacyjnego (Windows, macOS, Linux).
3. Uruchom instalator i zaznacz opcję "Add Python to PATH" (ważne!).
4. Po instalacji sprawdź, czy Python działa poprawnie, wpisując w terminalu:
   ```
   python --version
   ```
   Powinno wyświetlić się coś w stylu `Python 3.12.X`.

### Przygotowanie środowiska Python
1. Stwórz wirtualne środowisko (venv):
   ```
   python -m venv nazwa_venva
   ```
2. Aktywuj środowisko:
   - Windows:
     ```
     nazwa_venva\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```
     source nazwa_venva/bin/activate
     ```
3. Zainstaluj wymagane paczki:
   ```
   pip install -r requirements.txt
   ```

## 4. Dodawanie zmian

### a) Pobranie najnowszych zmian z chmury z gałęzi `main`
```bash
git switch main
git pull
```

### b) Tworzenie własnej gałęzi
Wszystkie zmiany / dodatki do kodu robimy na własnej gałęzi. Gałęzie powinny mieć nazwę:
```bash
feature/nazwa-zadania
```
Na przykład, jeśli pracujesz nad funkcją `preprocess_data`
```bash
git checkout -b feature/preprocess-data
```

### c) Zapisywanie zmian lokalnie
- Dodanie plików do zapisania:
```bash
git add .
```
- Zrobienie commitu (zapis zmiany):
```bash
git commit -m "Krótki opis zmian"
```
- Przykład:
```bash
git commit -m "Dodanie funkcji, która przerabia surowe dane na takie, które są gotowe do użycia w modelu"
```

### d) Wysyłanie zmian do chmury (push)
Gdy zakończysz etap pracy i chcesz wysłać zmiany na serwer:
```bash
git push origin nazwa-twojego-brancha
```
- Przykład:
```bash
git push origin feature/preprocess-data
```

### Podsumowanie
Krok po kroku:
1. `git pull origin main` — zawsze na początku pracy.
2. `git checkout -b feature/nazwa-zadania` — nowa gałęź dla Twojej pracy.
3. `git add . i git commit -m "opis"` — zapis zmian lokalnie.
4. `git push origin feature/nazwa-zadania` — wysyłka zmian do chmury.

Pamiętaj, że możesz zawsze pytać, jeśli coś będzie niejasne!

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

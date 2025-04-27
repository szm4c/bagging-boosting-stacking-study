# Instrukcja korzystania z repozytorium

## 1. Struktura repozytorium

W repozytorium znajdziesz kilka głównych katalogów:
- src/ - tutaj znajdują się wszystkie pliki z kodem źródłowym.
- notebooks/ - tutaj zapisujemy nasze notebooki eksperymentalne.
- data/ - dane potrzebne do pracy. folder
- tests/ - tym się nie przejmujcie :)
- requirements.txt - lista wszystkich bibliotek potrzebnych do uruchomienia projektu.

## 2. Klonowanie repozytorium lokalnie

Aby skopiować repozytorium na swój komputer odpal w terminalu:
```
git clone https://github.com/szm4c/bagging-boosting-stacking-study.git
```
Adres tego repo można też dostać klikając `<> Code` (zielony przycisk) -> `Local` -> `HTTPS` na głównej stronie repozytorium.

## 3. Przygotowanie środowiska Python pod pracę
- Pracujemy na Pythonie 3.12
Zalecam stworzenie wirtualnego środowiska (venv) u siebie lokalnie na komputerze:
```
python -m venv nazwa_venva
```
następnie należy je aktywować, żeby zainstować w nim potrzebne paczki z pliku `requirements.txt`
- Windows:
```
venv\Scripts\Activate.ps1
```
- Max/Linux:
```
tutaj nie wiem XD
```
Na koniec trzeba zainstalować paczki:
```
pip install -r requirements.txt
```

## 4. Dodawanie zmian
### a) Pobranie najnowszych zmian z chmury z gałęzi `main`
```
git switch main
git pull
```
### b) Tworzenie własnej gałęzi
Wszystkie zmiany / dodatki do kodu robimy na własnej gałęzi. Gałęzie powinny mieć nazwę:
```
feature/nazwa-zadania
```
Na przykład, jeśli pracujesz nad funkcją `preprocess_data`
```
git checkout -b feature/preprocess-data
```
### c) Zapisywanie zmian lokalnie
- dodanie plików do zapisania
```
git add .
```
- zrobienie commitu (zapis zmiany)
```
git commit -m "Krótki opis zmian"
```
- przykład
```
git commit -m "Dodanie funkcji, która przerabia surowe dane na takie, które są gotowe do użycia w modelu"
```
### d) Wysyłanie zmian do chmury (push)
Gdy zakończysz etap pracy i chcesz wysłać zmiany na serwer
```
git push nazwa-twojego-brancha
```
- przykład:
```
git push feature/preprocess-data
```
### Podsumowanie
Krok po kroku:
1. `git pull origin main` — zawsze na początku pracy.
2. `git checkout -b feature/nazwa-zadania` — nowa gałęź dla Twojej pracy.
3. `git add . i git commit -m "opis"` — zapis zmian lokalnie.
4. `git push origin feature/nazwa-zadania` — wysyłka zmian do chmury.
Pamiętaj, że możesz zawsze pytać, jeśli coś będzie niejasne!

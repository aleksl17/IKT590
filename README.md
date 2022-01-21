# IKT590
IKT590 Master's Thesis

# Environment Suggestions (Subject to Change)
Python 3.9.10

virtualenv

## Alx Setup (Subject to change):
Last ned Python 3.9.10 (all users / global install anbefalt)

Installer virtualenv  med pip: `python -m pip install --upgrade virtualenv`

Lag et virtualenv i `IKT590` git folderen med kommandoen `virtualenv .ikt590_virtualenv`

(".ikt590..." er virtualenv navnet. Kan kalles noe annet, men da må man legge til det i .gitignore)

Åpne terminal of your choice og naviger til `IKT590` folderen. Skriv `.\.ikt590_virtualenv\Script\activate`

Obs, den operasjonen krever mest sannsynelig endring i execution policy. Kjør `Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser` eller [les mer her.](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies)

Når du er i virtualenv kjør `python -m pip install -r requirements.txt`

For å kjøre scriptet benytt deg av `main.py` filen. Alt annet er moduler.

Foreløpig ligger det to VS Code launch.json configs. De burde funke for alle om man har fulgt guiden over. Bruk "Python: main" configen for å kjøre scriptet i VS Code.

Hvis VS Code ikke finner virtualenv interperteren. Trykk "ctrl + shift + p" og finn "Select Interperter". Her finn virtualenv python.exe under "%USERPROFILE%/IKT590/.ikt590_virtualenv/Scripts/python.exe".

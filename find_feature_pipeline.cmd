@echo off
echo ==========================================
echo SmartSOC - Production Feature Pipeline Search
echo ==========================================
echo.

echo [1] Searching all project files for model artifact references...
findstr /S /N /I /C:"model.joblib" *.py *.json *.yaml *.yml *.txt *.md 2>nul

echo.
echo [2] Searching for 6919 / final feature references...
findstr /S /N /I /C:"6919" /C:"final_features" /C:"feature_names" *.py *.json *.yaml *.yml *.txt *.md 2>nul

echo.
echo [3] Searching for preprocessing artifacts...
findstr /S /N /I /C:"tfidf" /C:"vectorizer" /C:"scaler" /C:"transform" *.py *.json *.yaml *.yml *.txt *.md 2>nul

echo.
echo [4] Searching model directory...
dir /S /B models\windows_apt_v4_1\production 2>nul

echo.
echo ==========================================
echo Search complete.
echo ==========================================
pause
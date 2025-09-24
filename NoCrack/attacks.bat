@echo off

if exist "..\env\Scripts\activate.bat" (
    call ..\env\Scripts\activate.bat
) else (
    call conda activate honeyvault
)
set "SCRIPT_DIR=%~dp0attacks\"

echo Running kl divergence attack
python "%SCRIPT_DIR%kl_divergence.py" --num 100

echo Running single password attack
python "%SCRIPT_DIR%single_password.py" --num 100

echo Running theoretically grounded attack
python "%SCRIPT_DIR%theoretically_grounded.py" --num 100

echo Running weak and strong encoding attacks
python "%SCRIPT_DIR%encoding.py" --num 100

echo Running password similarity attack
python "%SCRIPT_DIR%password_similarity.py" --num 100

echo Running hybrid attack
python "%SCRIPT_DIR%hybrid.py" --num 100

if exist "..\env\Scripts\activate.bat" (
    call deactivate
) else (
    call conda deactivate
)
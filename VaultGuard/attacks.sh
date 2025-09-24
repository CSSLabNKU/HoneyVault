#!/bin/bash

if [ -f "../env/bin/activate" ]; then
    source ../env/bin/activate
else
    conda activate honeyvault
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/attacks/"

echo "Running kl divergence attack"
python "${SCRIPT_DIR}kl_divergence.py" --num 100

echo "Running single password attack"
python "${SCRIPT_DIR}single_password.py" --num 100

echo "Running theoretically grounded attack"
python "${SCRIPT_DIR}theoretically_grounded.py" --num 100

echo "Running weak and strong encoding attacks"
python "${SCRIPT_DIR}encoding.py" --num 100

echo "Running password similarity attack"
python "${SCRIPT_DIR}password_similarity.py" --num 100

echo "Running adaptive extra attack"
python "${SCRIPT_DIR}adaptive_extra.py" --num 100

echo "Running adaptive hybrid attack"
python "${SCRIPT_DIR}adaptive_hybrid.py" --num 100

if [ -f "../env/bin/activate" ]; then
    deactivate
else
    conda deactivate
fi
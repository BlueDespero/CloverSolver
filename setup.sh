if [ ! -d "./venv" ]; then
  echo "Creating CloverSolver virtual env"
  python3 -m venv ./venv
fi

source ./venv/bin/activate

set -o pipefail
python3 -m pip install -r ./requirements.txt | { grep -v "already satisfied" || :; }

cs_path=$(pwd)
export PYTHONPATH="$cs_path:$PYTHONPATH"

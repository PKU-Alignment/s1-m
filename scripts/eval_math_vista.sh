set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../evaluation-test_time_scaling_law/MathVista" || exit 1

python eval.py
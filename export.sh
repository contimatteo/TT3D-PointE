###

# exit 0


GPU=0
PROMPT="test_t3bench_n1"

ROOT_DIR="/media/data2/mconti/TT3D"
OUT_DIR="${ROOT_DIR}/outputs/${PROMPT}"


###


CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
  --source-path "${OUT_DIR}/OpenAI-PointE/"

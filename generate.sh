###

# exit 0


GPU=0
PROMPT="test_t3bench_n1"

ROOT_DIR="/media/data2/mconti/TT3D"
PROMPT_DIR="${ROOT_DIR}/prompts"
OUT_DIR="${ROOT_DIR}/outputs/${PROMPT}"
PROMPT_FILE="${PROMPT_DIR}/${PROMPT}.txt"


###


CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/OpenAI-PointE/"
# --skip-existing  

###

exit 0


ROOT_DIR="/media/data2/mconti/TT3D"
OUT_DIR="${ROOT_DIR}/outputs"

GPU=0


###


CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
  --source-path "${OUT_DIR}/PointE/"

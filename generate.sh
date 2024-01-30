###
### Unibo CV-LAB Server run script 
###

exit 0

CUDA_VISIBLE_DEVICES=2 python3 tt3d_generate.py \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.v1.n2.txt \
  --out-path /media/data2/mconti/TT3D/models/PointE/outputs/

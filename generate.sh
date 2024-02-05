###
### Unibo CV-LAB Server run script 
###

exit 0

CUDA_VISIBLE_DEVICES=1 python3 tt3d_generate.py \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.t3bench.n1.txt \
  --out-path /media/data2/mconti/TT3D/outputs/PointE/
# --skip-existing  

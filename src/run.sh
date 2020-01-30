for model in oracle mf rmf
do
python main.py \
  $model \
  --eps 5.0 \
  --pow_list 0.5 1.0 2.0 3.0 4.0 \
  --dim 10 \
  --lam 1e-5 \
  --eta 1e-1 \
  --max_iters 1000 \
  --batch_size 12 \
  --iters 5
done

python visualize.py \
  --eps 5.0 \
  --pow_list 0.5 1.0 2.0 3.0 4.0

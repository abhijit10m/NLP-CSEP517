python train.py --data_path data/xsum_train.txt --val_data_path data/xsum_train.txt --embed_file None --n_val_batches 10 --model_path_prefix checkpoints/q2_0 --attn_type cosine
python test.py --model checkpoints/q2_0.10.pt --test_data_path data/xsum_test.txt --test_gen_path ../generations/RNN_cosine.txt
# Note, in this example:
#   - query images are NVS RENDERED images from gaussian-splatting TEST split
#   - reference images are REAL CAPTURED images from gaussian-splatting TRAIN split

ckpt_path=ckpt/CrossScore-v1.0.0.ckpt
data_dir=datadir/processed_training_ready/gaussian/map-free-reloc/res_540

for scene_name in s00076 s00231; do

  query_dir=$data_dir/$scene_name/test/ours_15000/renders
  reference_dir=$data_dir/$scene_name/train/ours_15000/gt
  
  python task/predict.py \
    trainer.devices=[0] \
    trainer.ckpt_path_to_load=ckpt/CrossScore-v1.0.0.ckpt \
    data.dataset.query_dir=$query_dir \
    data.dataset.reference_dir=$reference_dir \
    alias=$scene_name
done

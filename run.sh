

# data transform 
python data_transform.py \
--src_root_path=/home/hy/vscode/pcb_custom/Market-1501-v15.09.15  \
--dst_root_path=/home/hy/vscode/pcb_custom/datasets/Market1501 

##data transform  fro kaggle
!python data_transform.py \
--src_root_path=/kaggle/input/market1501/Market-1501-v15.09.15  \
--dst_root_path=/kaggle/working/datasets/Market1501




# data train 
 !python train.py \
 --dataset_path=/home/hy/vscode/pcb_custom/datasets/Market1501
##data train  fro kaggle
 !python train.py \
--dataset_path=/kaggle/working/datasets/Market1501
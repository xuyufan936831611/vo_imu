1、准备数据：
python data/prepare_train_data_3v1.py --dataset_dir=/home/omnisky/xuyufan/视觉里程计/dataset/KITTI/raw_data/kitti_raw_eigen/ --dataset_name='kitti_raw_eigen' --dump_root=eigen_data/02/ --seq_length=3 --img_width=416 --img_height=128 --num_threads=4
python data/prepare_train_data_3v1.py --dataset_dir=/home/omnisky/xuyufan/视觉里程计/dataset/KITTI/raw_data/kitti_raw_eigen/ --dataset_name='kitti_raw_eigen' --dump_root=eigen_data/03/ --seq_length=1 --img_width=416 --img_height=128 --num_threads=4

python data/prepare_train_data.py --dataset_dir=/home/omnisky/xuyufan/视觉里程计/dataset/KITTI/ --dataset_name='kitti_odom' --dump_root=/home/omnisky/xuyufan/视觉里程计/NEWVO(disp) --seq_length=3 --img_width=416 --img_height=128 --num_threads=4


2、训练：
python train.py --dataset_dir_l=/media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/omnisky_backup/xuyufan/VO/old_version/NEWVO_MASK/eigen_data/02 --dataset_dir_r=/media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/omnisky_backup/xuyufan/VO/old_version/NEWVO_MASK/eigen_data/03 --checkpoint_dir=checkpoints/ --img_width=416 --img_height=128 --batch_size=4
python train.py --dataset_dir_l=/media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/xuyufan/SVO_RNN_N_sequences/vo_dump_3/02 --dataset_dir_r=/media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/xuyufan/SVO_RNN_N_sequences/vo_dump_3/03 --checkpoint_dir=checkpoints/ --img_width=416 --img_height=128 --batch_size=4

3：tensorboard显示：
tensorboard --logdir=checkpoints/ --port=8888


4、PoseNet测试：
python test_kitti_pose.py --test_seq [sequence_id] --dataset_dir /path/to/KITTI/odometry/set/ --output_dir /path/to/output/directory/ --ckpt_file /path/to/pre-trained/model/file/

python test_kitti_pose.py --test_seq 9 --dataset_dir /home/omnisky/xuyufan/视觉里程计/dataset/KITTI/ --output_dir /home/omnisky/xuyufan/视觉里程计/code/UNDEEPVO/kitti_eval/pose_data/my_result/ --ckpt_file models/model-100280


python test_kitti_pose.py --test_seq 9 --dataset_dir /home/omnisky/xuyufan/视觉里程计/dataset/KITTI/ --output_dir /home/omnisky/xuyufan/视觉里程计/code/UNDEEPVO/kitti_eval/pose_data/ --output_dir /path/to/output/directory/ --ckpt_file models/model-100280


5、
python test_kitti_depth.py --dataset_dir /home/omnisky/xuyufan/视觉里程计/dataset/KITTI/raw_data/kitti_raw_eigen/ --output_dir depth_mask/ --ckpt_file checkpoints/model-191178


python kitti_eval/eval_depth.py --kitti_dir=/home/omnisky/xuyufan/视觉里程计/dataset/KITTI/raw_data/kitti_raw_eigen/ --pred_file=depth_3/model-147180.npy


6、验证(计算ATE)：
python3 kitti_eval/eval_pose.py --gtruth_dir=kitti_eval/pose_data/ground_truth/10/ --pred_dir=kitti_eval/pose_data/ours_results/10/

python kitti_eval/eval_pose.py --gtruth_dir=/home/omnisky/xuyufan/视觉里程计/old version/their_code_python2/UNDEEPVO/kitti_eval/pose_data/ground_truth/09/ --pred_dir=kitti_eval/pose_data/my_result/09/


python test_kitti_pose_all.py --dataset_dir /media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/xuyufan/备份/视觉里程计/dataset/KITTI/ --output_dir pose_data/  --ckpt_dir checkpoints
python kitti_eval/eval_pose_all.py --gtruth_dir=/home/omnisky/xuyufan/视觉里程计/old_version/their_code_python2/UNDEEPVO/kitti_eval/pose_data/ground_truth/ --pred_dir=pose_data/

python ./tools/evaluation_tools.py --func eval_odom --odom_result_dir /media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/xuyufan/SVO/pose_abs/model.latest/

python test_kitti_depth_all.py --dataset_dir /media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/xuyufan/备份/视觉里程计/dataset/KITTI/raw_data/kitti_raw_eigen/ --output_dir depth_deconv/ --ckpt_dir checkpoints
python LR_eval/evaluate_kitti_depth_all.py --gt_path=/media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/xuyufan/备份/视觉里程计/dataset/KITTI/raw_data/kitti_raw_eigen/ --predicted_disp_path=depth_mask --split=eigen --garg_crop

python LR_eval/evaluate_kitti_depth_all_and_display.py --gt_path=/media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/xuyufan/备份/视觉里程计/dataset/KITTI/raw_data/kitti_raw_eigen/ --split=eigen --garg_crop --output_dir=depth_Godard --output_dir_gt=depth_gt --predicted_disp_path=/media/omnisky/902177fe-4ad6-4003-be5e-0f539ffb197c/xuyufan/SVO/best_depth_LR


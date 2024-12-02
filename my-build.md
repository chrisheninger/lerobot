Leader: /dev/tty.usbmodem58FD0165421
Follower: /dev/tty.usbmodem58FD0164191

python lerobot/scripts/configure_motor.py \
 --port /dev/tty.usbmodem58FD0165421 \
 --brand feetech \
 --model sts3215 \
 --baudrate 1000000 \
 --ID 1

1: Present Position [2047]
2: Present Position [2050]
3: Present Position [2050]
4: Present Position [2046]
5: Present Position [2050]
6: Present Position [2049]

1: Present Position [2049]
2: Present Position [2050]
3: Present Position [2050]
4: Present Position [2046]
5: Present Position [2049]
6: Present Position [2050]

conda activate lerobot

HF_USER=$(huggingface-cli whoami | head -n 1)

python lerobot/scripts/control_robot.py teleoperate \
 --robot-path lerobot/configs/robot/so100.yaml \
 --robot-overrides '~cameras' \
 --display-cameras 0

python lerobot/scripts/control_robot.py teleoperate \
 --robot-path lerobot/configs/robot/so100.yaml

python lerobot/scripts/control_robot.py record \
 --robot-path lerobot/configs/robot/so100.yaml \
 --fps 30 \
 --root data \
 --repo-id chrisheninger/so100_test_run \
 --tags so100 tutorial \
 --warmup-time-s 5 \
 --episode-time-s 30 \
 --reset-time-s 10 \
 --push-to-hub 0 \
 --num-episodes 50

DATA_DIR=data python lerobot/scripts/train.py \
 dataset_repo_id=chrisheninger/so100_move_cucumber \
 policy=act_so100_real \
 env=so100_real \
 hydra.run.dir=outputs/train/act_so100_move_cucumber \
 hydra.job.name=act_so100_move_cucumber \
 device=mps

python lerobot/scripts/control_robot.py record \
 --robot-path lerobot/configs/robot/so100.yaml \
 --fps 30 \
 --root data \
 --repo-id chrisheninger/eval_act_so100_move_cucumber \
 --tags so100 tutorial eval \
 --warmup-time-s 5 \
 --episode-time-s 40 \
 --reset-time-s 10 \
 --num-episodes 50 \
 --push-to-hub 0 \
 -p outputs/train/act_so100_move_cucumber/checkpoints/070000/pretrained_model

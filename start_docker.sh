docker run -it \
--mount type=bind,source="$(pwd)",target=/tf/master_thesis \
--mount type=bind,source=$(pwd)/.cache/,target=/root/.cache \
-p 8888:8888 --gpus all \
--rm tensorflow/tensorflow:latest-gpu-jupyter

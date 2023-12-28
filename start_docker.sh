
USER=wphu

nvidia-docker run --rm -it \
    -v /home/${USER}:/root \
    -v ${PWD}:/workspace \
    -v /ssd/${USER}:/ssd/${USER} \
    -v /home/${USER}:/home/${USER} \
    -v /training/${USER}:/training/${USER} \
    --shm-size 8G \
    -p 8080:8080 \
    -p 8501:8501 \
    harbor.aibee.cn/auto_car/visualglm:lavis.v1.1 bash  # base minigpt4.v1.1 add sat install

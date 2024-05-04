docker build -t pytorch2106rl_playground_llm .
xhost + 
docker run --rm --name llm -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6012:6006 -dit -v `pwd`/project:/app -v /tmp/.X11-unix:/tmp/.X11-unix pytorch2106rl_playground_llm

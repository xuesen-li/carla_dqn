# carla_dqn

### System Requirement

- Ubuntu 20.04
- Nvidia Driver Version: 520.61.05
- Cuda 11.8
- cudNN 8.9.6

### Python Dependencies

- Python 3.8
- Tensorflow 2.13
- carla 0.9.13

Python dependencies can be install with `conda env create -f environment.yml`

### Run

- start carla: `./CarlaUE4.sh`
- train: `python carla_client_train.py`
- play: `python carla_client_play.py`



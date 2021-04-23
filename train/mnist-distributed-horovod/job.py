# description: train tensorflow CNN model on mnist data distributed via horovod

# For more information on using Horovod with TensorFlow, refer to Horovod documentation:
#
# * [Horovod with TensorFlow](https://github.com/horovod/horovod/blob/master/docs/tensorflow.rst)
# * [Horovod with Keras](https://github.com/horovod/horovod/blob/master/docs/keras.rst)

# imports
import os

from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment
from azureml.core.runconfig import MpiConfiguration

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(__file__).parent

# training script
source_dir = str(prefix.joinpath("src"))
script_name = "train.py"

# environment file
environment_file = str(prefix.joinpath("environment.yml"))

# azure ml settings
environment_name = "tensorflow-gpu-example"
experiment_name = "tensorflow-mnist-distributed"
compute_name = "gpu-8x-a100"

# create environment
env = Environment.from_conda_specification(environment_name, environment_file)
env.docker.enabled = True

# Experiment configuration
node_count=2 # number of nodes
process_count_per_node=8 # number of GPUs per node

# Env configuration option
env_option = 2
# build env
if env_option == 1:
    dockerfile_name=None
    env.docker.base_image = "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"
elif env_option == 2:
    dockerfile_name=None
    env.docker.base_image = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04"
elif env_option == 3:    
    dockerfile_name = 'Dockerfile-cuda11.1.1-cudnn8-devel-ubuntu18.04'        
    env.docker.base_image=None
    env.docker.base_dockerfile=open(dockerfile_name, "r").read()
else:
    raise Exception('Unsupported env_option')

# create distributed config
distr_config = MpiConfiguration(process_count_per_node=process_count_per_node, node_count=node_count)

# create arguments
args = ["--epochs", 10]

# create job config
src = ScriptRunConfig(
    source_directory=source_dir,
    script=script_name,
    arguments=args,
    compute_target=compute_name,
    environment=env,
    distributed_job_config=distr_config,
)

# submit job
if dockerfile_name:
    docker_config = dockerfile_name
else:
    docker_config = env.docker.base_image
run = Experiment(ws, experiment_name).submit(src, tags={'distr_config:': 'MpiConfiguration', 'node_count': node_count, 'process_count_per_node': process_count_per_node, 'docker_config': docker_config })
run.wait_for_completion(show_output=True)

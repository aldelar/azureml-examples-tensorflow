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
environment_name = "tf-gpu-horovod-example"
experiment_name = "tf-mnist-distributed-horovod-example"
compute_name = "gpu-8x-a100"

#
dockerfile_name = None

# create environment
env = Environment.from_conda_specification(environment_name, environment_file)
env.docker.enabled = True

# Option #1
# specify a GPU base image
#env.docker.base_image = (
    #"mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"
    #"mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04"
#)

# Option #2
# specify a Dockerfile
dockerfile_name = 'Dockerfile'
env.docker.base_image=None
env.docker.base_dockerfile=open(dockerfile_name, "r").read()

# Create a `ScriptRunConfig` to specify the training script & arguments, environment, and cluster to run on.
#
# Create an `MpiConfiguration` to run an MPI/Horovod job.
# Specify a `process_count_per_node` equal to the number of GPUs available per node of your cluster.

# create distributed config
process_count_per_node=8
node_count=2
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
run = Experiment(ws, experiment_name).submit(src, tags={'docker': docker_config, 'node_count': str(node_count), 'process_count_per_node': str(process_count_per_node) })
run.wait_for_completion(show_output=True)

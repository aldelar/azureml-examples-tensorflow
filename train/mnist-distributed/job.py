# description: train tensorflow CNN model on mnist data distributed via tensorflow

# Train a distributed TensorFlow job using the `tf.distribute.Strategy` API on Azure ML.
#
# For more information on distributed training with TensorFlow, refer [here](https://www.tensorflow.org/guide/distributed_training).

# imports
import os

from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment
from azureml.core.runconfig import TensorflowConfiguration

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
environment_name = "tf-gpu-example"
experiment_name = "tf-mnist-distributed"
compute_name = "gpu-8x-a100"

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
dockerfile_name = 'Dockerfile-cuda11.1.1-cudnn8-devel-ubuntu18.04'
env.docker.base_image=None
env.docker.base_dockerfile=open(dockerfile_name, "r").read()

# Create a `ScriptRunConfig` to specify the training script & arguments, environment, and cluster to run on.
#
# The training script in this example utilizes multi-worker distributed training of a Keras model using the `tf.distribute.Strategy` API,
# specifically `tf.distribute.experimental.MultiWorkerMirroredStrategy`. To run a multi-worker TensorFlow job on Azure ML, create a
# `TensorflowConfiguration`. Specify a `worker_count` corresponding to the number of nodes for your training job.
#
# In TensorFlow, the `TF_CONFIG` environment variable is required for training on multiple machines.
# Azure ML will configure and set the `TF_CONFIG` variable appropriately for each worker before executing your training script.
# You can access `TF_CONFIG` from your training script if you need to via `os.environ['TF_CONFIG']`.

# create distributed config
worker_count=2
distr_config = TensorflowConfiguration(worker_count=worker_count, parameter_server_count=0)

# create args
model_path = os.path.join("./outputs", "keras-model")

args = ["--epochs", 10, "--model-dir", model_path]

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
run = Experiment(ws, experiment_name).submit(src, tags={'distr_config:': 'TensorflowConfiguration', 'worker_count': worker_count, 'docker_config': docker_config })
run.wait_for_completion(show_output=True)

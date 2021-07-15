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

# azure ml settings
experiment_name = "tensorflow-mnist-distributed"
compute_name = "gpu-8x-a100"

# environment
env = Environment.get(workspace=ws, name="AzureML-tensorflow-2.4-ubuntu18.04-py37-cuda11-gpu").clone("tensorflow-2.4-gpu")

# Experiment configuration
node_count=2 # number of nodes
process_count_per_node=8 # number of GPUs per node
    
# create distributed config
distr_config = MpiConfiguration(process_count_per_node=process_count_per_node, node_count=node_count)

# create arguments
args = ["--epochs", 5]

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
run = Experiment(ws, experiment_name).submit(src, tags={'distr_config:': 'MpiConfiguration', 'node_count': str(node_count), 'process_count_per_node': str(process_count_per_node)})
run.wait_for_completion(show_output=True)

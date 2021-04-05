from pathlib import Path

from azureml.core import (
    Environment,
    Experiment,
    Run,
    ScriptRunConfig,
    Workspace,
)
from azureml.core.compute import AmlCompute, ComputeTarget


def find_or_create_compute_target(
    workspace,
    name,
    vm_size="Standard_D8_v3",
    min_nodes=0,
    max_nodes=1,
    idle_seconds_before_scaledown=900,
    vm_priority="lowpriority",
):

    if name in workspace.compute_targets:
        return ComputeTarget(workspace=workspace, name=name)
    else:
        config = AmlCompute.provisioning_configuration(
            vm_size=vm_size,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            vm_priority=vm_priority,
            idle_seconds_before_scaledown=idle_seconds_before_scaledown,
        )
        target = ComputeTarget.create(workspace, name, config)
        target.wait_for_completion(show_output=True)
    return target


workspace_config = 'config.json'
requirements_file = 'requirements.txt'
compute_target_name = 'my-compute'
experiment_name = 'simple-example'
script_path = 'simple/run.py'
script_args = ['--message', 'Howdy!']


# Authenticate with your AzureML Resource via its config.json file
ws = Workspace.from_config(workspace_config)

# The experiment in this workspace under which our runs will be grouped
# If an experiment with the given name doesn't exist, it will be created
exp = Experiment(ws, experiment_name)

# The compute cluster you want to run on and its settings.
# If it doesn't exist, it'll be created.
compute_target = find_or_create_compute_target(ws, compute_target_name)

# The Environment lets us define any dependencies needed to make our script run
env = Environment.from_pip_requirements("my-pip-env", requirements_file)

# A run configuration is how you define what youd like to run
# We give it the directory where our code is, the script we want to run, the environment, and the compute info
run_config = ScriptRunConfig(
    source_directory=Path(script_path).parent,
    script=Path(script_path).name,
    arguments=script_args,
    compute_target=compute_target,
    environment=env,
)

# Submit our configured run under our experiment
run = exp.submit(run_config)

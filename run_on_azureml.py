import itertools
from pathlib import Path

import fire
from azureml.core import (
    Environment,
    Experiment,
    ScriptRunConfig,
    Workspace,
)
from azureml.core.compute import AmlCompute, ComputeTarget


def find_or_create_compute_target(
        workspace,
        name,
        instance_type="Standard_D8_v3",
        min_nodes=0,
        max_nodes=10,
        idle_seconds_before_scaledown=900,
        vm_priority="lowpriority",
):
    if name in workspace.compute_targets:
        return ComputeTarget(workspace=workspace, name=name)
    else:
        config = AmlCompute.provisioning_configuration(
            vm_size=instance_type,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            vm_priority=vm_priority,
            idle_seconds_before_scaledown=idle_seconds_before_scaledown,
        )
        target = ComputeTarget.create(workspace, name, config)
        target.wait_for_completion(show_output=True)
    return target


def main(
        workspace_config: str = "config.json",
        experiment_name: str = "azureml-demo-exp",
        compute_target_name: str = "my-compute",
        requirements_file: str = "requirements.txt",
        script_path: str = "simple_example/run.py",
        delete_compute: bool = False,
        **kwargs,
):
    ws = Workspace.from_config(workspace_config)
    exp = Experiment(ws, experiment_name)
    target = find_or_create_compute_target(ws, compute_target_name)
    env = Environment.from_pip_requirements("my-pip-env", requirements_file)
    args = list(itertools.chain(*[(f"--{n}", f"{v}") for n, v in kwargs.items()]))
    run_config = ScriptRunConfig(
        source_directory=Path(script_path).parent,
        script=Path(script_path).name,
        arguments=args,
        compute_target=target,
        environment=env,
    )
    run = exp.submit(run_config)
    run.wait_for_completion(show_output=True)
    if delete_compute:
        target.delete()


if __name__ == "__main__":
    fire.Fire(main)

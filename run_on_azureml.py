from argparse import ArgumentParser
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
    instance_type="Standard_D8_v3",
    min_nodes=0,
    max_nodes=10,
    idle_seconds_before_scaledown=900,
    vm_priority='lowpriority'
):
    if name in workspace.compute_targets:
        return ComputeTarget(workspace=workspace, name=name)
    else:
        config = AmlCompute.provisioning_configuration(
            vm_size=instance_type,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            vm_priority=vm_priority,
            idle_seconds_before_scaledown=idle_seconds_before_scaledown
        )
        target = ComputeTarget.create(workspace, name, config)
        target.wait_for_completion(show_output=True)
    return target


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--workspace_config', type=str, default='config.json')
    parser.add_argument('--experiment_name', type=str, default='azureml-demo-exp')
    parser.add_argument('--compute_target_name', type=str, default='my-compute')
    parser.add_argument('--requirements_file', type=str, default='requirements.txt')
    parser.add_argument('--script_path', type=str, default='simple_example/run.py')
    parser.add_argument('--delete_compute', action='store_true', default=False)
    args, extras = parser.parse_known_args(args)
    args.script_args = extras
    return args


def main(args):
    ws = Workspace.from_config(args.workspace_config)
    exp = Experiment(ws, args.experiment_name)
    target = find_or_create_compute_target(ws, args.compute_target_name)
    env = Environment.from_pip_requirements('my-pip-env', args.requirements_file)
    run_config = ScriptRunConfig(
        source_directory=Path(args.script_path).parent,
        script=Path(args.script_path).name,
        arguments=args.script_args,
        compute_target=target,
        environment=env
    )
    run = exp.submit(run_config)
    run.wait_for_completion(show_output=True)
    if args.delete_compute:
        target.delete()


if __name__ == '__main__':
    main(parse_args())

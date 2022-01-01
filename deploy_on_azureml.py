from argparse import ArgumentParser
from pathlib import Path

from azureml.core import Experiment, Run
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice
from azureml.core.workspace import Workspace


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("--workspace_config", type=str, default="config.json")
    parser.add_argument("--requirements_file", type=str, default="requirements.txt")
    parser.add_argument("--experiment_name", type=str, default="test-exp")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--model_artifact_path", type=str, default="logs/saved_model")
    parser.add_argument("--score_file", type=str, default="keras_example/score.py")
    parser.add_argument("--model_name", type=str, default="test-keras-model")
    parser.add_argument("--service_name", type=str, default="test-keras-service")
    return parser.parse_args(args)


def main(args):

    # Init workspace object from azureml workspace resource you've created
    ws = Workspace.from_config(args.workspace_config)

    # Point to an experiment
    experiment = Experiment(ws, name=args.experiment_name)

    run = Run(experiment, args.run_id)

    # Register your best run's model
    model = run.register_model(
        model_name=args.model_name, model_path=args.model_artifact_path
    )

    # Create an environment based on requirements
    env = Environment.from_pip_requirements("my-pip-env", args.requirements_file)

    # Get inference config to configure API's behavior
    inference_config = InferenceConfig(
        source_directory=Path(args.score_file).parent,
        entry_script=Path(args.score_file).name,
        environment=env,
        enable_gpu=False,
    )

    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1, memory_gb=1, description="A dummy model that returns a set message"
    )

    service = Model.deploy(
        workspace=ws,
        name=args.service_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config,
    )

    service.wait_for_deployment(True)
    return service


if __name__ == "__main__":
    service = main(parse_args())

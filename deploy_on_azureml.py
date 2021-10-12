from pathlib import Path

import fire
from azureml.core import Experiment, Run
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice
from azureml.core.workspace import Workspace


def main(
        workspace_config: str = "config.json",
        requirements_file: str = "requirements.txt",
        experiment_name: str = "test-exp",
        run_id: str = None,
        model_artifact_path: str = "logs/saved_model",
        score_file: str = "keras_example/score.py",
        model_name: str = "test-keras-model",
        service_name: str = "test-keras-service",
):
    # Init workspace object from azureml workspace resource you've created
    ws = Workspace.from_config(workspace_config)

    # Point to an experiment
    experiment = Experiment(ws, name=experiment_name)

    run = Run(experiment, run_id)

    # Register your best run's model
    model = run.register_model(model_name=model_name, model_path=model_artifact_path)

    # Create an environment based on requirements
    env = Environment.from_pip_requirements("my-pip-env", requirements_file)

    # Get inference config to configure API's behavior
    inference_config = InferenceConfig(
        source_directory=Path(score_file).parent,
        entry_script=Path(score_file).name,
        environment=env,
        enable_gpu=False,
    )

    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1, memory_gb=1, description="A dummy model that returns a set message"
    )

    service = Model.deploy(
        workspace=ws,
        name=service_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config,
    )

    service.wait_for_deployment(True)
    return service


if __name__ == "__main__":
    fire.Fire(main)

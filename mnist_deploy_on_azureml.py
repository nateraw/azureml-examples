from pathlib import Path

from azureml.core import Experiment, Run
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice
from azureml.core.workspace import Workspace


if __name__ == '__main__':
    config_filepath = './config.json'
    experiment_name = 'mnist-example'
    run_id = 'mnist-example_1617615007_1ee478d9'
    model_artifact_path = 'logs/lightning_logs/version_0/checkpoints/epoch=4-step=8594.ckpt'
    requirements_file = 'requirements.txt'
    score_file = 'mnist_run/score.py'
    model_name = 'mnist-model'
    service_name = 'mnist-service'

    # Init workspace object from azureml workspace resource you've created
    ws = Workspace.from_config(config_filepath)

    # Point to an experiment
    experiment = Experiment(ws, name=experiment_name)

    run = Run(experiment, run_id)

    # # Register your best run's model
    model = run.register_model(
        model_name=model_name,
        model_path=model_artifact_path
    )

    # Create an environment based on requirements
    env = Environment.from_pip_requirements("my-pip-env", requirements_file)


    # Get inference config to configure API's behavior
    inference_config = InferenceConfig(
        source_directory=Path(score_file).parent,
        entry_script=Path(score_file).name,
        environment=env,
        enable_gpu=False,
    )

    # Get deployment config which outlines your API service's specs
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1,
        memory_gb=1,
        description='A dummy model that returns a set message'
    )

    # Deploy the service
    service = Model.deploy(
        workspace=ws,
        name=service_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config
    )

    service.wait_for_deployment(True)
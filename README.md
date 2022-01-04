# Azure Machine Learning Examples

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/nateraw/azureml-examples/main.svg)](https://results.pre-commit.ci/latest/github/nateraw/azureml-examples/main)
[![CI testing](https://github.com/nateraw/azureml-examples/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/nateraw/azureml-examples/actions/workflows/ci-testing.yml)

# Blogposts:

- [Create Your First Azure ML Workspace](https://nateraw.com/2021/04/azureml-create/)
- [Running Your First Python Script on Azure ML](https://nateraw.com/2021/04/azureml-basic/)

# Examples

## Simple Example

### Train

```
python run_on_azureml.py \
  --experiment test-exp \
  --script simple_example/run.py \
  --message Howdy!
```

### Deploy

```
python deploy_on_azureml.py \
  --run_id test-exp_1618105880_601b57cd \
  --experiment_name test-exp \
  --model_artifact_path logs/message.txt \
  --score_file simple_example/score.py \
  --model_name message-model \
  --service_name message-service
```

### Predict on Endpoint

```
python simple_example/score.py --endpoint <YOUR ENDPOINT>
```

## Keras

### Train

```
python run_on_azureml.py \
  --experiment test-exp \
  --script keras_mnist_example/train.py \
  --max_epochs 10 \
  --batch_size 64
```

### Deploy

In the Azure ML Portal, navigate to the run you're happy with, and copy its Run ID. Note this is not just an int like "
Run 5", but a longer unique identifier. Pass this to the `--run_id` flag of `deploy_on_azure.py`. Your Run ID should
look similar to the one below, but will not be the same.

```
python deploy_on_azureml.py \
  --run_id test-exp_1618103328_5e55046d \
  --experiment_name test-exp \
  --model_artifact_path logs/saved_model \
  --score_file keras_mnist_example/score.py \
  --model_name test-keras-model \
  --service_name test-keras-service
```

### Predict from Endpoint

```
python keras_mnist_example/score.py --endpoint <YOUR ENDPOINT>
```

## PyTorch Lightning Example

### Train

```
python run_on_azureml.py \
  --experiment test-exp \
  --script lightning_mnist_example/train.py \
  --max_epochs 10 \
  --batch_size 64 \
  --default_root_dir logs/
```

### Deploy

```
 python deploy_on_azureml.py \
  --run_id test-exp_1618104960_e34305b3 \
  --experiment_name test-exp \
  --model_artifact_path logs/lightning_logs/version_0/checkpoints/epoch=9-step=8599.ckpt \
  --score_file lightning_mnist_example/score.py \
  --model_name test-lit-model \
  --service_name test-lit-service
```

### Predict from Endpoint

```
python lightning_mnist_example/score.py --endpoint <YOUR ENDPOINT>
```

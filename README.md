# Azure Machine Learning Examples

## Simple Example

#### Train

Run a script that simply writes a file `./logs/message.txt` given argument `--message` that we pass manually to the script with value `"Howdy!"`

```
python simple_run_on_azureml.py
```

#### Deploy

Reference the written `./logs/message.txt` in the run you submitted to register a model on AzureML. Deploy it to an endpoint that just returns the message we provided.

You'll have to manually update the run_id.

```
python simple_deploy_on_azureml.py
```

#### Use Deployment to Predict

Update `endpoint` in `simple_client.py` with the endpoint of your deployment from the previous script. You can find this easily in the AzureML Portal. Then, run:

```
python simple_client.py
```

The output should be:

```
Response {"message": "Howdy!", "input_data": "blah"}
```
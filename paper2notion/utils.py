import datetime
import os

import mlflow


def turn_on_mlflow(
    experiment_name: str,
    port=30001,
):
    # mlflow.set_tracking_uri(f"{base_url}:{port}")
    print(f'mlflow tracking uri: {os.getenv("MLFLOW_TRACKING_URI")}')

    # if not no_time:
    # run_name += datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


def create_run_name(config):
    # check if config is a dictionary
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")

    run_name = datetime.datetime.now().strftime("%m%d-%H%M%S")
    run_name += f"_p{config['prompt_num']}_{config['model']}_t{config['temperature']}"

    return run_name

# pytorch-lightining-mlflow-example
An example of PyTorch Lightning &amp; MLflow logging sessions for a simple CNN usecase.

## Setup 

In order to run the code a simple strategy would be to create a pyhton 3.8 [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) and run the following:

```bash
$ conda create -f conda.yaml
$ conda activate pl-mlflow
```

## Setup a MLflow project

A MLflow project is basically a collection of specifications of the analytical project that allow MLflow to rerun any pipeline in a given environment in order to allow reproducibility of any experiments. If run in a git repository it also tracks the last committed code version for any run; for more information take a look [here](https://www.mlflow.org/docs/latest/projects.html).

In this use case example we need to define two files:

1. `conda.yaml` : this file defines the conda environment where the code should be run
2. `MLProject` : this files defines the basic properties of the project, i.e. name, envirnoment and entry points (see docs)

Once the project is setup with this files the code can be executed running 

```bash
$ mlflow run $PROJECT_FOLDER
```

In place of `$PROJECT_FOLDER` one could also use a github url to the project of interest.

> **REMARK**: at each run MLflow will try to create a new environment unless the user does not specifies not to. In order to use the current environment use flag `--no-conda`.

To access run results use the visual interface

```bash
$ mlflow ui --backend-store-uri $LOG_FOLDER
```

where `$LOG_FOLDER` is expected to host the MLflow logging directory.




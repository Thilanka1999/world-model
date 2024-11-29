from argparse import ArgumentParser
import shutil
import os
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.inputs import TrainingInput


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if os.path.exists("sagemaker/src/mt_pipe"):
        shutil.rmtree("sagemaker/src/mt_pipe")
    if os.path.exists("sagemaker/src/src"):
        shutil.rmtree("sagemaker/src/src")
    if os.path.exists("sagemaker/src/expm"):
        shutil.rmtree("sagemaker/src/expm")

    shutil.copytree("mt_pipe", "sagemaker/src/mt_pipe")
    shutil.copytree("src", "sagemaker/src/src")
    shutil.copytree("expm", "sagemaker/src/expm")

    estimator = PyTorch(
        role="arn:aws:iam::880561552723:role/SagemakerRole",
        instance_count=1,
        instance_type="ml.g4dn.2xlarge",
        output_path="s3://sagemaker-out/",
        source_dir="sagemaker/src",
        entry_point="main.py",
        py_version="py310",
        framework_version="2.1.0",
        hyperparameters={
            "replica-size": "1",
            "mock-batch-count": "1",
            "mock-epoch-count": "1",
            "config": args.config,
        },
        distribution={"pytorchddp": {"enabled": "true"}},
    )

    s3_input = TrainingInput(s3_data="s3://ie-datasets", input_mode="FastFile")
    inputs = {"training": s3_input}

    try:
        estimator.fit(inputs)
    except Exception as e:
        shutil.rmtree("sagemaker/src/mt_pipe")
        shutil.rmtree("sagemaker/src/src")
        shutil.rmtree("sagemaker/src/expm")
        raise e

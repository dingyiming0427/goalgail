
import os.path as osp
import os

USE_GPU = False

USE_TF = True

AWS_REGION_NAME = "us-west-1"

if USE_GPU:
    DOCKER_IMAGE = "dementrock/rllab3-shared-gpu-cuda80"
else:
    DOCKER_IMAGE = "dementrock/rllab3-shared"

DOCKER_LOG_DIR = "/tmp/expt"

AWS_S3_PATH = "s3://rllab-s3-yiming/rllab_goal/experiments"

AWS_CODE_SYNC_S3_PATH = "s3://rllab-s3-yiming/rllab_goal/code"

ALL_REGION_AWS_IMAGE_IDS = {
    "ap-northeast-1": "ami-c42689a5",
    "ap-northeast-2": "ami-865b8fe8",
    "ap-south-1": "ami-ea9feb85",
    "ap-southeast-1": "ami-c74aeaa4",
    "ap-southeast-2": "ami-0792ae64",
    "eu-central-1": "ami-f652a999",
    "eu-west-1": "ami-8c0a5dff",
    "sa-east-1": "ami-3f2cb053",
    "us-east-1": "ami-de5171c9",
    "us-east-2": "ami-e0481285",
    # "us-west-1": "ami-efb5ff8f",
    # "us-west-1": "ami-afc985cf",
    "us-west-1": "ami-0bd5a554f953caee5",
    "us-west-2": "ami-53903033",
}

ALL_SUBNET_INFO = {
    "ap-northeast-1c": {
        "SubnetID": "subnet-5bda4200",
        "Groups": "sg-5b021b22"
    },
    "ap-northeast-1d": {
        "SubnetID": "subnet-8a4178a2",
        "Groups": "sg-5b021b22"
    },
    "ap-northeast-1a": {
        "SubnetID": "subnet-9b7f50d2",
        "Groups": "sg-5b021b22"
    },
    "ap-northeast-2a": {
        "SubnetID": "subnet-af742cc7",
        "Groups": "sg-7625d61c"
    },
    "ap-northeast-2c": {
        "SubnetID": "subnet-06a22d4a",
        "Groups": "sg-7625d61c"
    },
    "ap-south-1a": {
        "SubnetID": "subnet-df75e1b7",
        "Groups": "sg-38136453"
    },
    "ap-south-1b": {
        "SubnetID": "subnet-5ddd1911",
        "Groups": "sg-38136453"
    },
    "ap-southeast-1a": {
        "SubnetID": "subnet-05763b4c",
        "Groups": "sg-962e45ef"
    },
    "ap-southeast-1c": {
        "SubnetID": "subnet-0f3d2349",
        "Groups": "sg-962e45ef"
    },
    "ap-southeast-1b": {
        "SubnetID": "subnet-66fda001",
        "Groups": "sg-962e45ef"
    },
    "ap-southeast-2a": {
        "SubnetID": "subnet-88fe84ef",
        "Groups": "sg-493a5930"
    },
    "ap-southeast-2c": {
        "SubnetID": "subnet-f37d85ab",
        "Groups": "sg-493a5930"
    },
    "ap-southeast-2b": {
        "SubnetID": "subnet-fa295eb3",
        "Groups": "sg-493a5930"
    },
    "eu-central-1a": {
        "SubnetID": "subnet-fce4b497",
        "Groups": "sg-5cbed131"
    },
    "eu-central-1b": {
        "SubnetID": "subnet-bf25acc2",
        "Groups": "sg-5cbed131"
    },
    "eu-central-1c": {
        "SubnetID": "subnet-0770124a",
        "Groups": "sg-5cbed131"
    },
    "eu-west-1b": {
        "SubnetID": "subnet-7b56f31d",
        "Groups": "sg-209c845a"
    },
    "eu-west-1a": {
        "SubnetID": "subnet-0cbb5c56",
        "Groups": "sg-209c845a"
    },
    "eu-west-1c": {
        "SubnetID": "subnet-edc76da5",
        "Groups": "sg-209c845a"
    },
    "sa-east-1a": {
        "SubnetID": "subnet-82af9fe5",
        "Groups": "sg-b08ccbd6"
    },
    "sa-east-1c": {
        "SubnetID": "subnet-ab2723f3",
        "Groups": "sg-b08ccbd6"
    },
    "us-east-1b": {
        "SubnetID": "subnet-dd11fdba",
        "Groups": "sg-c7fa238e"
    },
    "us-east-1a": {
        "SubnetID": "subnet-7700172a",
        "Groups": "sg-c7fa238e"
    },
    "us-east-1f": {
        "SubnetID": "subnet-5e79ca51",
        "Groups": "sg-c7fa238e"
    },
    "us-east-1d": {
        "SubnetID": "subnet-45b9620f",
        "Groups": "sg-c7fa238e"
    },
    "us-east-1e": {
        "SubnetID": "subnet-61724b5e",
        "Groups": "sg-c7fa238e"
    },
    "us-east-1c": {
        "SubnetID": "subnet-1f031230",
        "Groups": "sg-c7fa238e"
    },
    "us-east-2a": {
        "SubnetID": "subnet-0b003962",
        "Groups": "sg-4fbf1427"
    },
    "us-east-2c": {
        "SubnetID": "subnet-cb2daa86",
        "Groups": "sg-4fbf1427"
    },
    "us-east-2b": {
        "SubnetID": "subnet-473c443c",
        "Groups": "sg-4fbf1427"
    },
    "us-west-1b": {
        "SubnetID": "subnet-d5244e8e",
        "Groups": "sg-33b0894a"
    },
    "us-west-1c": {
        "SubnetID": "subnet-64f5ab03",
        "Groups": "sg-33b0894a"
    },
    "us-west-2a": {
        "SubnetID": "subnet-34e1367c",
        "Groups": "sg-1fd32562"
    },
    "us-west-2b": {
        "SubnetID": "subnet-e6328480",
        "Groups": "sg-1fd32562"
    },
    "us-west-2c": {
        "SubnetID": "subnet-5dadd706",
        "Groups": "sg-1fd32562"
    }
}

INSTANCE_TYPE_INFO = {  #this prices are orientative.
    "c4.large": dict(price=0.105, vCPU=2),
    "c4.xlarge": dict(price=0.209, vCPU=4),
    "c4.2xlarge": dict(price=0.419, vCPU=8),
    "c4.4xlarge": dict(price=0.838, vCPU=16),
    "c4.8xlarge": dict(price=1.00, vCPU=36),
    "m4.large": dict(price=0.1, vCPU=2),
    "m4.xlarge": dict(price=0.5, vCPU=4),
    "m4.2xlarge": dict(price=0.5, vCPU=8),
    "m4.4xlarge": dict(price=0.8, vCPU=16),
    "m4.10xlarge": dict(price=2.394, vCPU=40),
    "m4.16xlarge": dict(price=1.5, vCPU=64),
    "g2.2xlarge": dict(price=0.65, vCPU=8),
}

AWS_IMAGE_ID = ALL_REGION_AWS_IMAGE_IDS[AWS_REGION_NAME]

if USE_GPU:
    AWS_INSTANCE_TYPE = "g2.2xlarge"
else:
    AWS_INSTANCE_TYPE = "c4.2xlarge"

ALL_REGION_AWS_KEY_NAMES = {
    "ap-northeast-1": "rllab-ap-northeast-1",
    "ap-northeast-2": "rllab-ap-northeast-2",
    "ap-south-1": "rllab-ap-south-1",
    "ap-southeast-1": "rllab-ap-southeast-1",
    "ap-southeast-2": "rllab-ap-southeast-2",
    "eu-central-1": "rllab-eu-central-1",
    "eu-west-1": "rllab-eu-west-1",
    "sa-east-1": "rllab-sa-east-1",
    "us-east-1": "rllab-us-east-1",
    "us-east-2": "rllab-us-east-2",
    "us-west-1": "rllab-us-west-1",
    "us-west-2": "rllab-us-west-2"
}

AWS_KEY_NAME = ALL_REGION_AWS_KEY_NAMES[AWS_REGION_NAME]

AWS_SPOT = True

AWS_SPOT_PRICE = '0.5'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_IAM_INSTANCE_PROFILE_NAME = "rllab"

AWS_SECURITY_GROUPS = ["rllab-sg"]

ALL_REGION_AWS_SECURITY_GROUP_IDS = {
    "ap-northeast-1": [
        "sg-5b021b22"
    ],
    "ap-northeast-2": [
        "sg-7826d512"
    ],
    "ap-south-1": [
        "sg-38136453"
    ],
    "ap-southeast-1": [
        "sg-b22a41cb"
    ],
    "ap-southeast-2": [
        "sg-933d5eea"
    ],
    "eu-central-1": [
        "sg-b7bed1da"
    ],
    "eu-west-1": [
        "sg-209c845a"
    ],
    "sa-east-1": [
        "sg-da8dcabc"
    ],
    "us-east-1": [
        "sg-fbf22bb2"
    ],
    "us-east-2": [
        "sg-dc3b0fb7"
    ],
    "us-west-1": [
        "sg-33b0894a"
    ],
    "us-west-2": [
        "sg-faeb6b84"
    ]
}

AWS_SECURITY_GROUP_IDS = ALL_REGION_AWS_SECURITY_GROUP_IDS[AWS_REGION_NAME]

FAST_CODE_SYNC_IGNORES = [
    ".git",
    "data",
    "src",
    ".idea",
    ".pods",
    "tests",
    "examples",
    "docs",
    ".idea",
    ".DS_Store",
    ".ipynb_checkpoints",
    "blackbox",
    "blackbox.zip",
    "*.pyc",
    "*.ipynb",
    "scratch-notebooks",
    "conopt_root",
    "private/key_pairs",
]

FAST_CODE_SYNC = True


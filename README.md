# aws-sagemaker-custom-inf-issue
Issues when using [Detectron2/DensePose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) on AWS SageMaker

## Case1: using PyTorchModel by gzip
* reference: https://sagemaker.readthedocs.io/en/v2.32.1/frameworks/pytorch/using_pytorch.html
* result:

## PyTorchModel by Docker
* reference: https://github.com/aws/amazon-sagemaker-examples-community/blob/215215eb25b40eadaf126d055dbb718a245d7603/bring-your-own-container/pytorch_extending_our_containers/pytorch_extending_our_containers.ipynb
* result:
  
## PyTorchModel by ModelBuilder
* reference: https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-modelbuilder-creation.html
* result:

## Reference
* [SageMaker PyTorch Model](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html#pytorch-model)
* [Deploy a Trained PyTorch Model](https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_deploy.html)

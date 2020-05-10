# Ensemble Model to Train XRay Dataset

You can download dataset in [NIH Chest X-rays](https://www.kaggle.com/nih-chest-xrays/data) dataset

## Train & Test model
To train model, you can run the below script
```shell script
python train.py --model densenet --input path/of/input/data
```
You can choose one of the following models 
- densenet
- inceptionv3
- xception
- inception_resnet_v2

In each model you run, the weight is stored in `model.{model_name}.h5`

## Test model

You can download pretrain model `inception_resnet_v2` from this link [inception_resnet_v2](https://drive.google.com/open?id=1MYRQW9-8dhbxbWcpg7UUE-ZIMJfXBI0-)

After download, you can run the following script to test
```shell script
python test.py --image 'path/to/your/image' --model inception
```
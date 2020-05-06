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

In each model you run, the weight is stored in `model.{model_name}.h5`

If you train all models above, you can choose model_name is `ensemble` to calculate mean result of all model.

## Evaluate model

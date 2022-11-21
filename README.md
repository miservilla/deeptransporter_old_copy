# AI-predictor of  properties of transporters in the human placenta and lactating mammary epithelium.![image](https://github.com/vinash85/deeptransporter/deeptransporter.prelims.png)
 

*Authors: Avinash Sahu

## Description 

DeepTransporter is an AI predictor that employs structural features of transporters and substrates to predict substrate of any given transporter. We intend to model kinetic characterics of each substrate-transporter pair using the model. 
The repository holds suites of the software generated for finding predicting properties of substrate. Right now, we have tested the preliminary version of DeepTransporter, which is focused on predicting hexose transporters from its structural features. 


The pytorch package is based on [tutorials](https://cs230-stanford.github.io/project-starter-code.html).

## Requirements

We recommend using python3 and a virtual env. See instructions [here](https://cs230-stanford.github.io/project-starter-code.html).

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.




## Quickstart (~10 min)


1. __Your first experiment__ We created a `experiments` directory. It contains a file `params.json` which sets the hyperparameters for the experiment. It looks like
```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 10,
    ...
}
```
For every new experiment, you will need to create a new directory under `experiments` with a similar `params.json` file.
 datasets_tsne_list.txt contains dataset information, including its location etc. that looks like 


```
prefix  dataset_type    train_optimizer_mask    data_dir        tsne    types   data_augmentation
""      "tcga"  "[1,1,1]"       "./experiments/"     1 ['train','val']  [0,0]
```


2. __Train__ your experiment. Simply run
```
python train.py  --data_dir  ./experiments/datasets_tsne_list.txt --model_dir ./config/params.json 
```
It will instantiate a model and train it on the training set following the hyperparameters specified in `params.json`. It will also evaluate some metrics on the validation set. It will store the results in  ./tensorboardLog/outputdir.  Every train is run a different time-tagged outputdir is created. E.g. 20221113-141311 was created when I run last time.


3. __Evaluation on the test set__ Once you've run many experiments and selected your best model and hyperparameters based on the performance on the validation set, you can finally evaluate the performance of your model on the test set. Run
```
python evaluate.py  --data_dir  ./datasets_test_list.txt --model_dir ./tensorboardLog/20221113-141311/. --restore_file ./tensorboardLog/20221113-141311/epoch-20.pth.tar  --output_dir ./tensorboardLog/20221113-141311/eval_dir
```


## Guidelines for more advanced use

We recommend reading through `train.py` to get a high-level overview of the training loop steps:
- loading the hyperparameters for the experiment (the `params.json`)
- loading the training and validation data
- creating the model, loss_fn and metrics
- training the model for a given number of epochs by calling `train_and_evaluate(...)`

Once you get the high-level idea, depending on your dataset, you might want to modify
- `model/net.py` to change the neural network, loss function and metrics
- `model/data_loader.py` to suit the data loader to your specific needs
- `train.py` for changing the optimizer
- `train.py` and `evaluate.py` for some changes in the model or input require changes here

Once you get something working for your dataset, feel free to edit any part of the code to suit your own needs.

## Resources

- [PyTorch documentation](http://pytorch.org/docs/0.3.0/)
- [Tutorials](http://pytorch.org/tutorials/)
- [Tutorials of package](https://cs230-stanford.github.io/project-starter-code.html)
- [PyTorch warm-up](https://github.com/jcjohnson/pytorch-examples)

[SIGNS]: https://drive.google.com/file/d/1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC/view?usp=sharing

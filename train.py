import torch
from utils import preprocessing, parameter_parser_train, train
from src import Model


if __name__ == '__main__':

    args = parameter_parser_train()
    dataset = preprocessing.Dataset(args)
    model = Model(dataset)
    model.train()
    train(dataset, model, args)
    #  Load weights
    model.load_state_dict(torch.load('data/textGenerator_model.pt'))
    print("model saved")

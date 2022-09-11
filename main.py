import torch
import os
from utils import preprocessing, parameter_parser, predict, train
from src import Model


if __name__ == '__main__':

    args = parameter_parser()
    dataset = preprocessing.Dataset(args)
    model = Model(dataset)
    if args.load_model == True:
        if os.path.exists(args.model):

            model.eval()
            model.load_state_dict(torch.load(args.model))
            predict(dataset, model, text='царица')

    else:
        model.train()
        train(dataset, model, args)
        print(predict(dataset, model, text='царица'))
        # Load weights
        model.load_state_dict(torch.load('data/textGenerator_model.pt'))

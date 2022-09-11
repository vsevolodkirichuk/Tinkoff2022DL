import argparse

def parameter_parser_train():

    parser = argparse.ArgumentParser(description="Text Generation")
    parser.add_argument('--length', type=int, default=5)
    parser.add_argument('--input_dir', type=str, default='stdin')
    parser.add_argument("--input", dest="model", type=str, default='data/textGenerator_model.pt')

    return parser.parse_args()

def parameter_parser_generate():

    parser = argparse.ArgumentParser(description="Text Generation")
    parser.add_argument('--length', type=int, default=5)
    parser.add_argument("--prefix", dest="prefix", type=str)
    parser.add_argument("--model", dest="model", type=str, default='data/textGenerator_model.pt')

    return parser.parse_args()


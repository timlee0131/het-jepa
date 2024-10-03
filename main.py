import argparse
from experiments import trainer, tuner

def get_args():
    parser = argparse.ArgumentParser(description="Heterogeneous Graph JEPA")

    parser.add_argument('-m', '--mode', choices=['train', 'tune', 'test'], type=str, default='train', help='Mode: train, tune, test')
    
    parser.add_argument('-d', '--dataset', choices=['cora', 'pubmed', 'citeseer'], type=str, default='cora', help='dataset to use')

    return parser.parse_args()

def main():
    args = get_args()

    if args.mode == 'tune':
        tuner.driver(args.dataset)
    else:   # train by default
        trainer.driver(args.dataset)

if __name__ == '__main__':
    main()
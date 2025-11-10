from torchvision import datasets, transforms
from gilg_utils.general import save_pickle
from argparse import ArgumentParser
from os import makedirs, path

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        help='Filepath where input data will be saved.',
        required=True
    )
    parser.add_argument(
        '--output',
        dest='output',
        help='Filepath where output data will be saved.',
        required=True
    )
    args = parser.parse_args()

    # Download Data
    makedirs(path.dirname(args.output),exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(args.input, download=False, train=True, transform=transform)
    save_pickle(trainset,args.output)

if __name__ == '__main__':
    main()

from torchvision import datasets, transforms
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--output',
        dest='output',
        help='Filepath where output data will be saved.',
        required=True
    )
    args = parser.parse_args()

    # Download Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(args.output, download=True, train=True, transform=transform)

if __name__ == '__main__':
    main()

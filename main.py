import argparse
from utils.config import *
from agents import *
import torch

def main():
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--experiment', type=str, default='pcb_p6')
    parser.add_argument('--save_path', type=str, default='./experiments')
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke'])
    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch_size')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help='FC params learning rate')
    parser.add_argument('--epochs', default=60, type=int,
                        help='The number of epochs to train')
    parser.add_argument('--share_conv', action='store_true')
    parser.add_argument('--stripes', type=int, default=6)
    args = parser.parse_args()
    print(args)

    # Fix random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    # Make saving directory
    save_dir_path = os.path.join(args.save_path, args.experiment)
    print(save_dir_path)
    os.makedirs(save_dir_path, exist_ok=True)


if __name__ == "__main__":
    main()

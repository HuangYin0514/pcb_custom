import argparse
from agents import *
import torch
import torch.nn as nn
from graphs.models import build_model
import torch.optim as optim
from torch.optim import lr_scheduler
from agents import *
import datasets
import os


def main(args):

    # Fix random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    # Make saving directory
    save_dir_path = os.path.join(args.save_path, args.experiment)
    print(save_dir_path)
    os.makedirs(save_dir_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = datasets.getMarket1501DataLoader(
        args.dataset, args.batch_size, 'train', shuffle=True, augment=True)

    model = build_model(args.experiment, num_classes=6, num_stripes=args.stripes,
                        share_conv=args.share_conv, return_features=False)

    criterion = nn.CrossEntropyLoss()

    # Finetune the net
    optimizer = optim.SGD([
        {'params': model.backbone.parameters(), 'lr': args.learning_rate / 10},
        {'params': model.local_conv.parameters() if args.share_conv else model.local_conv_list.parameters(),
         'lr': args.learning_rate},
        {'params': model.fc_list.parameters(), 'lr': args.learning_rate}
    ], momentum=0.9, weight_decay=5e-4, nesterov=True)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Use multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    input = torch.randn(4, 3, 384, 128)
    print([k.shape for k in model(input)])

    agent = PCBAgent(args=args, model=model, optimizer=optimizer,
                     scheduler=scheduler, criterion=criterion)
    agent.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--experiment', type=str, default='PCB_p6')
    parser.add_argument('--save_path', type=str, default='./experiments')
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke'])
    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch_size')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help='FC params learning rate')
    parser.add_argument('--epochs', default=60, type=int,
                        help='The number of epochs to train')
    parser.add_argument('--share_conv', default=True, action='store_true')
    parser.add_argument('--stripes', type=int, default=6)
    args = parser.parse_args()

    print(args)

    main(args)
    # torch.cuda.empty_cache()

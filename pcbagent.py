import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from .agents.baseagent import BaseAgent
from .utils import *
import time

cudnn.benchmark = True


class PCBAgent(BaseAgent):
    def __init__(self, args, model, optimizer, scheduler, criterion):
        super().__init__(args)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.args = args

    def run(self):
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        start_time = time.time()
        # Logger instance
        logger = utils.Logger(save_dir_path)
        logger.info('-' * 10)
        logger.info(vars(args))


if __name__ == "__main__":

    agent = PCBAgent(args=None, model=None,
                     optimizer=None, scheduler=None)
    # agent.run()
    print(agent)

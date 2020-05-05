import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from agents.baseagent import BaseAgent

cudnn.benchmark = True


class PCBAgent(BaseAgent):
    def __init__(self, args, model, optimizer, scheduler):
        super().__init__(args)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

if __name__ == "__main__":

    agent = PCBAgent(args=None, model=None,
                      optimizer=None, scheduler=None)
    # agent.run()
    print(agent)

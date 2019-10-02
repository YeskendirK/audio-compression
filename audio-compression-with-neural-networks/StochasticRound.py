import torch
#from .module import Module

class StochasticRound(nn.Module):

    def __init__(self, inplace):
        super(StochasticRound, self).__init__()
        self.train = True
        self.inplace = inplace
        self.epsilon = torch.Tensor()
        #self.output = None
        #self.gradInput = None

    def updateOutput(self, input):
        if self.inplace:
            self.output.set_(input)
        else:
            self.output.resize_as_(input).copy_(input)

        self.epsilon = self.epsilon.double()
        self.output = self.output.double()
        input = input.double()

        if self.train:
            self.epsilon = input - torch.floor(input)
            self.output = torch.floor(input) + torch.bernoulli(self.epsilon)

            self.output = self.ouput.cuda()
        else:
            self.output = torch.floor(input) + torch.ge(self.epsilon, 0.5)

            self.output = self.output.cuda()

        return self.output


    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput
        return self.gradInput


    def __tostring__(self):
        return str.format('%s', type(self))
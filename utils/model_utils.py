import torch
def init_model(model):
    def init_all(model, init_func, *params, **kwargs):
        for p in model.parameters():
            init_func(p, *params, **kwargs)
    init_all(model, torch.nn.init.normal_, mean=0., std=1)

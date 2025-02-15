import torch


def load_public_model(cfg):
    if cfg.Model.model_name == 'P5':
        from models.public_models.P5 import P5
        model = P5(**cfg.Model)
        print('success to init P5')
    else:
        raise NotImplementedError
    return model


def load_model(cfg):
    if cfg.Model.model_type == 'public':
        model = load_public_model(cfg)
    else:
        raise NotImplementedError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    return model


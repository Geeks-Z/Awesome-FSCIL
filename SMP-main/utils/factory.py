import importlib

def get_model(model_name, args):
    name = model_name.lower()
    if name == "asp":
        from models.asp import Learner
        return Learner(args)
    elif name.startswith('px'):
        model = importlib.import_module('models.' + name)
        return model.Learner(args)
    else:
        assert 0
    


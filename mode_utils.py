import pickle

from cnn_model import build_model


def save_model(model, path='model.pkl'):
    state = []
    for layer in model:
        if hasattr(layer, 'get_state'):
            state.append(layer.get_state())
        else:
            state.append(None)
    with open(path, 'wb') as f:
        pickle.dump(state, f)


def load_model(path='model.pkl'):
    state = None
    with open(path, 'rb') as f:
        state = pickle.load(f)
    model = build_model()
    for layer, s in zip(model, state):
        if s is not None and hasattr(layer, 'set_state'):
            layer.set_state(s)
    return model

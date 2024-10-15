# EpyNN/epynn/settings.py


# HYPERPARAMETERS SETTINGS
se_hPars = {
    # Schedule learning rate
    'learning_rate': 0.001,
    'schedule': 'exp_decay',
    'decay_k': 1e-3,
    'cycle_epochs': 0,
    'cycle_descent': 0,
    # Tune activation function
    'ELU_alpha': 0.01,
    'LRELU_alpha': 0.01,
    'softmax_temperature': 1,
}
"""Hyperparameters dictionary settings.

Set hyperparameters for model and layer.
"""

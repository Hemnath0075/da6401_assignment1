import argparse
import wandb
import tensorflow as tf
from model import NeuralNetwork

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network with Weights & Biases tracking')
    parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname', help='WandB project name')
    parser.add_argument('-we', '--wandb_entity', type=str, default='myname', help='WandB entity name')
    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='fashion_mnist', help='Dataset to use')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('-l', '--loss', type=str, choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy', help='Loss function')
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='sgd', help='Optimizer')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help='Momentum for optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.5, help='Beta for RMSprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5, help='Beta1 for Adam and Nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5, help='Beta2 for Adam and Nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6, help='Epsilon for optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help='Weight decay for optimizers')
    parser.add_argument('-w_i', '--weight_init', type=str, choices=['random', 'Xavier'], default='random', help='Weight initialization')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=4, help='Number of neurons per hidden layer')
    parser.add_argument('-a', '--activation', type=str, choices=['identity', 'sigmoid', 'tanh', 'ReLU'], default='sigmoid', help='Activation function')
    return parser.parse_args()

# Train model
def train(args):
    # Initialize wandb
    wandb.login()
    config = vars(args)
    wandb.init(config=config, entity=args.wandb_entity, project=args.wandb_project)

    # Load dataset
    if args.dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
        raise ValueError('Choose from mnist or fashion_mnist ...')

    # Preprocess data
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    # Initialize model
    model = NeuralNetwork(
        input_neurons=784,
        hidden_layers=args.num_layers,
        output_neurons=10,
        weight_init=args.weight_init,
        activation=args.activation,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2
    )

    # Train model
    model.train(x_train, y_train, args.epochs, args.batch_size)

if __name__ == '__main__':
    args = parse_args()
    train(args)

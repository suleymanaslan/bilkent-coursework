import numpy as np
from matplotlib import pyplot as plt


def plot_lra_random_weight_losses(linear_regressor_ann, input_dim, output_dim, train1_x, train1_nb_examples, train1_y):
    min_lra_loss = np.inf
    random_weights = np.arange(-10,11)
    fig = plt.figure()
    fig.set_facecolor('w')
    for i in random_weights:
        linear_regressor_ann.set_weights(i.reshape((output_dim, input_dim)))
        lra_output = linear_regressor_ann.forward(train1_x.reshape((train1_nb_examples, input_dim, output_dim)))
        lra_loss = np.mean(linear_regressor_ann.loss(train1_y))
        if lra_loss < min_lra_loss:
            min_lra_loss = lra_loss
        plt.scatter(i, lra_loss, color="blue")
    plt.title('Loss for Linear regressor ANN')
    plt.xlabel('weights')
    plt.ylabel('loss')
    print(f"Minimum loss:{min_lra_loss:.2f}")
    plt.show()


def plot_tla_random_weight_losses(two_layer_ann, input_dim, output_dim, nb_of_hiddenunits, train1_x, train1_nb_examples, train1_y, randomize_first_layer):
    random_weights = np.arange(-10,11)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in random_weights:
        for j in random_weights:
            if randomize_first_layer:
                two_layer_ann.set_weights_1(np.array([i, j]).reshape((nb_of_hiddenunits, input_dim)))
            else:
                two_layer_ann.set_weights_2(np.array([i, j]).reshape((output_dim, nb_of_hiddenunits)))
            tla_output = two_layer_ann.forward(train1_x.reshape((train1_nb_examples, input_dim, output_dim)))
            tla_loss = np.mean(two_layer_ann.loss(train1_y))
            ax.scatter(i, j, tla_loss, color="blue")
    plt.title('Loss for Two Layer ANN')
    if randomize_first_layer:
        ax.set_xlabel('weights_1')
        ax.set_ylabel('weights_1')
    else:
        ax.set_xlabel('weights_2')
        ax.set_ylabel('weights_2')
    ax.set_zlabel('loss')
    plt.show()


def train_lra(nb_of_epochs, batch_size, learning_rate, linear_regressor_ann, train1_x, train1_y, input_dim, output_dim, train1_nb_examples, train1_uniform_x_samples):
    fig = plt.figure()
    min_lra_loss = np.inf
    for epoch in range(nb_of_epochs):
        for i in range(train1_nb_examples//batch_size):
            linear_regressor_ann.forward(train1_x[i*batch_size:i*batch_size+batch_size].reshape((batch_size, input_dim, output_dim)))
            linear_regressor_ann.loss(train1_y[i*batch_size:i*batch_size+batch_size])
            linear_regressor_ann.backward(learning_rate)
        lra_output = linear_regressor_ann.forward(train1_x.reshape((train1_nb_examples, input_dim, output_dim)))
        lra_loss = np.mean(linear_regressor_ann.loss(train1_y))
        print(f"Epoch:{epoch+1}, Linear regressor ANN loss:{lra_loss:.4f}")
        plt.scatter(train1_x, train1_y)
        lra_output = linear_regressor_ann.forward(train1_uniform_x_samples.reshape((train1_nb_examples, input_dim, output_dim)))
        plt.plot(train1_uniform_x_samples, lra_output.reshape((train1_nb_examples, 1)), linewidth=3)
        plt.title(f'Linear regressor ANN, Epoch:{epoch+1}, Training Set, Loss:{lra_loss:.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'gif/lra_{epoch+1:03d}.png')
        plt.close()
        if min_lra_loss - lra_loss > 1e-5:
            min_lra_loss = lra_loss
        else:
            print("Stopped training")
            plt.scatter(train1_x, train1_y)
            lra_output = linear_regressor_ann.forward(train1_uniform_x_samples.reshape((train1_nb_examples, input_dim, output_dim)))
            plt.plot(train1_uniform_x_samples, lra_output.reshape((train1_nb_examples, 1)), linewidth=3)
            plt.title(f'Linear regressor ANN, Epoch:{epoch+1}, Training Set, Loss:{lra_loss:.4f}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig('output/lra_train.png')
            plt.close()
            break


def evaluate_lra(linear_regressor_ann, x, nb_examples, input_dim, output_dim, y, mode_str):
    lra_output = linear_regressor_ann.forward(x.reshape((nb_examples, input_dim, output_dim)))
    lra_loss = np.mean(linear_regressor_ann.loss(y))
    lra_loss_std = np.std(linear_regressor_ann.loss(y))
    print(f"Linear regressor ANN, {mode_str} set loss:{lra_loss:.4f}, std:{lra_loss_std:.4f}")
    return lra_loss


def plot_lra_evaluation(linear_regressor_ann, x, input_dim, output_dim, y, mode_str, lra_loss, train_uniform_x_samples, train_nb_examples, label):
    fig = plt.figure()
    fig.set_facecolor('w')
    plt.scatter(x, y)
    lra_output = linear_regressor_ann.forward(train_uniform_x_samples.reshape((train_nb_examples, input_dim, output_dim)))
    plt.plot(train_uniform_x_samples, lra_output.reshape((train_nb_examples, 1)), linewidth=3)
    plt.title(f'Linear regressor ANN, {mode_str} Set, Loss:{np.mean(lra_loss):.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'output/lra_{label}_curve.png')
    plt.show()

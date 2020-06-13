import numpy as np
from matplotlib import pyplot as plt

import neural_net
import utils


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


def train_lra(hyperparams, linear_regressor_ann, train_x, train_y, input_dim, output_dim, train_nb_examples, train_uniform_x_samples, label="train"):
    learning_rate, nb_of_epochs, batch_size = hyperparams
    fig = plt.figure()
    min_lra_loss = np.inf
    for epoch in range(nb_of_epochs):
        for i in range(train_nb_examples//batch_size):
            linear_regressor_ann.forward(train_x[i*batch_size:i*batch_size+batch_size].reshape((batch_size, input_dim, output_dim)))
            linear_regressor_ann.loss(train_y[i*batch_size:i*batch_size+batch_size])
            linear_regressor_ann.backward(learning_rate)
        lra_output = linear_regressor_ann.forward(train_x.reshape((train_nb_examples, input_dim, output_dim)))
        lra_loss = np.mean(linear_regressor_ann.loss(train_y))
        print(f"Epoch:{epoch+1}, Linear regressor ANN loss:{lra_loss:.4f}")
        plt.scatter(train_x, train_y)
        lra_output = linear_regressor_ann.forward(train_uniform_x_samples.reshape((train_nb_examples, input_dim, output_dim)))
        plt.plot(train_uniform_x_samples, lra_output.reshape((train_nb_examples, 1)), linewidth=3)
        plt.title(f'Linear regressor ANN, Epoch:{epoch+1}, Training Set, Loss:{lra_loss:.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'gif/lra_{epoch+1:03d}.png')
        plt.close()
        if min_lra_loss - lra_loss > 1e-5:
            min_lra_loss = lra_loss
        else:
            print("Stopped training")
            plt.scatter(train_x, train_y)
            lra_output = linear_regressor_ann.forward(train_uniform_x_samples.reshape((train_nb_examples, input_dim, output_dim)))
            plt.plot(train_uniform_x_samples, lra_output.reshape((train_nb_examples, 1)), linewidth=3)
            plt.title(f'Linear regressor ANN, Epoch:{epoch+1}, Training Set, Loss:{lra_loss:.4f}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(f'output/lra_{label}.png')
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
    plt.savefig(f'output/lra_{label}.png')
    plt.show()


def train_tla(data_ix, train_x, train_y, input_dim, output_dim, train_nb_examples, train_uniform_x_samples):
    if data_ix == 1:
        lr_config = {2: 8e-3, 4: 1e-2, 8: 1e-2, 16: 5e-3}
        epoch_config = {2: 5500, 4: 8000, 8: 7500, 16: 9000}
        batchsize_config = {2: 2, 4: 2, 8: 2, 16: 3}
        activation_config = {2: "sigmoid", 4: "sigmoid", 8: "sigmoid", 16: "sigmoid"}
        loss_config = {2: "mse", 4: "mse", 8: "mse", 16: "mse"}
        momentum_config = {2: 0.75, 4: 0.75, 8: 0.9, 16: 0.6}
        stop_loss_config = {2: 0.05795, 4: 0.02025, 8: 0.02045, 16: 0.02065}
        plot_color = {2: "red", 4: "cyan", 8: "magenta", 16: "black"}
    elif data_ix == 2:
        lr_config = {2: 9e-3, 4: 9e-3, 8: 1e-2, 16: 9e-3}
        epoch_config = {2: 7500, 4: 6500, 8: 8000, 16: 30000}
        batchsize_config = {2: 3, 4: 3, 8: 2, 16: 2}
        activation_config = {2: "sigmoid", 4: "sigmoid", 8: "sigmoid", 16: "sigmoid"}
        loss_config = {2: "mse", 4: "mse", 8: "mse", 16: "mse"}
        momentum_config = {2: 0.4, 4: 0.4, 8: 0.5, 16: 0.3}
        stop_loss_config = {2: 0.28005, 4: 0.14305, 8: 0.05975, 16: 0.05915}
        plot_color = {2: "red", 4: "cyan", 8: "magenta", 16: "black"}

    trained_nets = []
    anim_files = []

    for nb_of_hiddenunits in (2, 4, 8, 16):
        np.random.seed(550)
        learning_rate = lr_config[nb_of_hiddenunits]
        nb_of_epochs = epoch_config[nb_of_hiddenunits]
        batch_size = batchsize_config[nb_of_hiddenunits]

        two_layer_ann = neural_net.TwoLayerANN(nb_of_hiddenunits, 
                                               activation_function=activation_config[nb_of_hiddenunits], 
                                               loss_function=loss_config[nb_of_hiddenunits], 
                                               use_momentum=True, momentum_factor=momentum_config[nb_of_hiddenunits])


        fig = plt.figure()
        print(f"Training two layer ANN with {nb_of_hiddenunits} units")
        for epoch in range(nb_of_epochs):
            for i in range(train_nb_examples//batch_size):
                two_layer_ann.forward(train_x[i*batch_size:i*batch_size+batch_size].reshape((batch_size, input_dim, output_dim)))
                two_layer_ann.loss(train_y[i*batch_size:i*batch_size+batch_size])
                two_layer_ann.backward(learning_rate)
            tla_output = two_layer_ann.forward(train_x.reshape((train_nb_examples, input_dim, output_dim)))
            tla_loss = np.mean(two_layer_ann.loss(train_y))
            if (data_ix == 1 and (epoch == 0 or (epoch+1) % 500 == 0)) or (data_ix == 2 and epoch == 0 or (epoch+1) % (1500 if nb_of_hiddenunits == 16 else 500) == 0):
                print(f"Epoch:{epoch+1}, Two layer ANN loss:{tla_loss:.4f}")
                plt.scatter(train_x, train_y)
                tla_output = two_layer_ann.forward(train_uniform_x_samples.reshape((train_nb_examples, input_dim, output_dim)))
                plt.plot(train_uniform_x_samples, tla_output.reshape((train_nb_examples, 1)), 
                         color=plot_color[nb_of_hiddenunits], linewidth=3)
                plt.title(f'Two layer ANN ({nb_of_hiddenunits} units), Epoch:{epoch+1}, Training Set, Loss:{tla_loss:.4f}')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.savefig(f'gif/tla_{nb_of_hiddenunits}_{epoch+1:05d}.png')
                plt.close()
            if tla_loss < stop_loss_config[nb_of_hiddenunits]:
                print(f"Stopped training, Epoch:{epoch+1}, Two layer ANN loss:{tla_loss:.4f}")
                plt.scatter(train_x, train_y)
                tla_output = two_layer_ann.forward(train_uniform_x_samples.reshape((train_nb_examples, input_dim, output_dim)))
                plt.plot(train_uniform_x_samples, tla_output.reshape((train_nb_examples, 1)), 
                         color=plot_color[nb_of_hiddenunits], linewidth=3)
                plt.title(f'Two layer ANN ({nb_of_hiddenunits} units), Epoch:{epoch+1}, Training Set, Loss:{tla_loss:.4f}')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.savefig(f'output/tla_{nb_of_hiddenunits}_train{"_2" if data_ix == 2 else ""}.png')
                plt.close()
                break

        anim_file = f'gif/tla_{nb_of_hiddenunits}_training{"_2" if data_ix == 2 else ""}.gif'
        utils.create_animation(anim_file, f'gif/tla_{nb_of_hiddenunits}_*.png', fps=4 if data_ix == 1 else 8)

        trained_nets.append(two_layer_ann)
        anim_files.append(anim_file)
    return trained_nets, anim_files


def evaluate_tla(trained_nets, train_x, train_y, test_x, test_y, train_nb_examples, test_nb_examples, train_uniform_x_samples, input_dim, output_dim, label=""):
    ann_hidden_units = [2, 4, 8, 16]
    plot_color = {2: "red", 4: "cyan", 8: "magenta", 16: "black"}
    for i in range(4):
        two_layer_ann = trained_nets[i]
        tla_output = two_layer_ann.forward(train_x.reshape((train_nb_examples, input_dim, output_dim)))
        tla_loss = np.mean(two_layer_ann.loss(train_y))
        tla_loss_std = np.std(two_layer_ann.loss(train_y))
        print(f"Two layer ANN, {ann_hidden_units[i]} units, training set loss:{tla_loss:.4f}, std:{tla_loss_std:.4f}")
        
        fig = plt.figure()
        fig.set_facecolor('w')
        plt.scatter(train_x, train_y)
        tla_output = two_layer_ann.forward(train_uniform_x_samples.reshape((train_nb_examples, input_dim, output_dim)))
        plt.plot(train_uniform_x_samples, tla_output.reshape((train_nb_examples, 1)), 
                 color=plot_color[ann_hidden_units[i]], linewidth=3)
        plt.title(f'Two layer ANN, {ann_hidden_units[i]} units, Training Set, Loss:{np.mean(tla_loss):.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'output/tla_{ann_hidden_units[i]}_train_curve{label}.png')
        plt.show()
        
        tla_output = two_layer_ann.forward(test_x.reshape((test_nb_examples, input_dim, output_dim)))
        tla_loss = np.mean(two_layer_ann.loss(test_y))
        tla_loss_std = np.std(two_layer_ann.loss(test_y))
        print(f"Two layer ANN, {ann_hidden_units[i]} units, test set loss:{tla_loss:.4f}, std:{tla_loss_std:.4f}")
        
        fig = plt.figure()
        fig.set_facecolor('w')
        plt.scatter(test_x, test_y)
        tla_output = two_layer_ann.forward(train_uniform_x_samples.reshape((train_nb_examples, input_dim, output_dim)))
        plt.plot(train_uniform_x_samples, tla_output.reshape((train_nb_examples, 1)), 
                 color=plot_color[ann_hidden_units[i]], linewidth=3)
        plt.title(f'Two layer ANN, {ann_hidden_units[i]} units, Test Set, Loss:{np.mean(tla_loss):.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'output/tla_{ann_hidden_units[i]}_test_curve{label}.png')
        plt.show()


def plot_tla_curves(trained_nets, train_x, train_y, train_uniform_x_samples, train_nb_examples, input_dim, output_dim, loc="upper left", label=""):
    plot_color = {2: "red", 4: "cyan", 8: "magenta", 16: "black"}
    fig = plt.figure()
    fig.set_facecolor('w')
    plt.scatter(train_x, train_y)

    two_layer_ann = trained_nets[0]
    tla_output = two_layer_ann.forward(train_uniform_x_samples.reshape((train_nb_examples, input_dim, output_dim)))
    plt.plot(train_uniform_x_samples, tla_output.reshape((train_nb_examples, 1)), label='2 Units', 
                 color=plot_color[2], linewidth=3)

    two_layer_ann = trained_nets[1]
    tla_output = two_layer_ann.forward(train_uniform_x_samples.reshape((train_nb_examples, input_dim, output_dim)))
    plt.plot(train_uniform_x_samples, tla_output.reshape((train_nb_examples, 1)), label='4 Units', 
                 color=plot_color[4], linewidth=6)

    two_layer_ann = trained_nets[2]
    tla_output = two_layer_ann.forward(train_uniform_x_samples.reshape((train_nb_examples, input_dim, output_dim)))
    plt.plot(train_uniform_x_samples, tla_output.reshape((train_nb_examples, 1)), label='8 Units', 
                 color=plot_color[8], linewidth=3)

    two_layer_ann = trained_nets[3]
    tla_output = two_layer_ann.forward(train_uniform_x_samples.reshape((train_nb_examples, input_dim, output_dim)))
    plt.plot(train_uniform_x_samples, tla_output.reshape((train_nb_examples, 1)), label='16 Units', 
                 color=plot_color[16], linewidth=3)

    leg = plt.legend(loc=loc)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)

    plt.title(f'Two layer ANNs with different number of hidden units')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'output/tla_all_curves{label}.png')
    plt.show()

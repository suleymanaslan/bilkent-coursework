import numpy as np


class LinearRegressorANN:
    def __init__(self, input_dim=1, output_dim=1):
        self.w1 = np.array(np.random.normal(size=(output_dim, input_dim)))
        self.b1 = np.full((output_dim, 1), 0.01)
    
    def set_weights(self, new_weights):
        self.w1 = new_weights
    
    def forward(self, inputs):
        self.x = inputs
        self.nb_of_input = self.x.shape[0]
        
        self.wpb = np.matmul(self.w1, self.x) + self.b1
        return self.wpb
    
    def loss(self, yt):
        self.err = yt.reshape((self.nb_of_input, 1)) - self.wpb.reshape((self.nb_of_input, 1))
        self.errsqr = self.err**2
        return self.errsqr
    
    def backward(self, learning_rate):
        derr = (2 * self.err)
        dyt = 1 * derr
        self.dwpb = -1 * derr
        dwmx = (1 * self.dwpb).reshape((self.nb_of_input, 1, 1))
        db1 = 1 * self.dwpb
        dw1 = np.matmul(dwmx, np.transpose(self.x, axes=(0, 2, 1)))
        dx = np.matmul(np.transpose(self.w1), dwmx) 
        
        self.w1 = self.w1 - learning_rate * np.mean(dw1, axis=0)
        self.b1 = self.b1 - learning_rate * np.mean(db1, axis=0)


class TwoLayerANN:
    def __init__(self, units, input_dim=1, output_dim=1, 
                 activation_function="relu", 
                 loss_function="mse", 
                 use_momentum=False, momentum_factor=0.9):
        self.w1 = np.random.normal(size=(units, input_dim)) * np.sqrt(2.0/input_dim)
        self.b1 = np.full((units, 1), 0.01)
        
        self.w2 = np.random.normal(size=(output_dim, units)) * np.sqrt(2.0/units)
        self.b2 = np.full((output_dim, 1), 0.01)
        
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.use_momentum = use_momentum
        self.momentum_factor = momentum_factor
        self.v1, self.v2, self.v3, self.v4 = 0, 0, 0, 0
    
    def set_weights_1(self, new_weights):
        self.w1 = new_weights
    
    def set_weights_2(self, new_weights):
        self.w2 = new_weights
    
    def forward(self, inputs):        
        self.x = inputs
        self.nb_of_input = self.x.shape[0]
        
        self.wmx = np.matmul(self.w1, self.x)
        self.wpb = self.wmx + self.b1
        self.act = self.hidden_activation()
        self.wmr = np.matmul(self.w2, self.act);
        self.wpb2 = self.wmr + self.b2
        return self.wpb2
    
    def hidden_activation(self):
        if self.activation_function == "relu":
            self.sigmoid = None
            self.lrelu = None
            self.relu = np.maximum(self.wpb, np.zeros_like(self.wpb))
            return self.relu
        elif self.activation_function == "sigmoid":
            self.relu = None
            self.lrelu = None
            self.sigmoid = 1.0 / (1 + np.exp(-self.wpb))
            return self.sigmoid
        elif self.activation_function == "lrelu":
            self.relu = None
            self.sigmoid = None
            self.lrelu_cons = 0.01
            self.lrelu = np.where(self.wpb > 0, self.wpb, self.lrelu_cons * self.wpb)
            return self.lrelu
    
    def loss(self, yt):
        if self.loss_function == "mse":
            self.abserr = None
            self.err = yt.reshape((self.nb_of_input, 1)) - self.wpb2.reshape((self.nb_of_input, 1))
            self.errsqr = self.err**2
            return self.errsqr
        elif self.loss_function == "mae":
            self.errsqr = None
            self.err = yt.reshape((self.nb_of_input, 1)) - self.wpb2.reshape((self.nb_of_input, 1))
            self.abserr = np.abs(self.err)
            return self.abserr
    
    def backward_loss(self):
        if self.loss_function == "mse":
            derr = (2 * self.err)
            dyt = 1 * derr
            self.dwpb2 = -1 * derr
            return self.dwpb2
        elif self.loss_function == "mae":
            derr = np.where(self.err > 0, 1, -1)
            dyt = 1 * derr
            self.dwpb2 = -1 * derr
            return self.dwpb2
    
    def backward_hidden_activation(self):
        if self.activation_function == "relu":
            self.dwpb = np.where(self.wpb > 0, 1 * self.dact, 0)
            return self.dwpb
        elif self.activation_function == "sigmoid":
            self.dwpb = ((1 - self.sigmoid) * self.sigmoid) * self.dact
            return self.dwpb
        elif self.activation_function == "lrelu":
            self.dwpb = np.where(self.wpb > 0, 1 * self.dact, self.lrelu_cons * self.dact)
            return self.dwpb
    
    def backward(self, learning_rate):
        self.dwpb2 = self.backward_loss()
        
        dwmr = (1 * self.dwpb2).reshape((self.nb_of_input, 1, 1))
        db2 = 1 * self.dwpb2
        dw2 = np.matmul(dwmr, np.transpose(self.act, axes=(0, 2, 1)))
        self.dact = np.matmul(np.transpose(self.w2), dwmr)
        
        self.dwpb = self.backward_hidden_activation()
        
        dwmx = 1 * self.dwpb
        db1 = 1 * self.dwpb
        
        dw1 = np.matmul(dwmx, np.transpose(self.x, axes=(0, 2, 1)))
        dx = np.matmul(np.transpose(self.w1), dwmx)
        
        if self.use_momentum:
            self.v1 = self.momentum_factor * self.v1 - learning_rate * np.mean(dw1, axis=0)
            self.w1 = self.w1 + self.v1
            self.v2 = self.momentum_factor * self.v2 - learning_rate * np.mean(db1, axis=0)
            self.b1 = self.b1 + self.v2
            self.v3 = self.momentum_factor * self.v3 - learning_rate * np.mean(dw2, axis=0)
            self.w2 = self.w2 + self.v3
            self.v4 = self.momentum_factor * self.v4 - learning_rate * np.mean(db2, axis=0)
            self.b2 = self.b2 + self.v4
        else:
            self.w1 = self.w1 - learning_rate * np.mean(dw1, axis=0)
            self.b1 = self.b1 - learning_rate * np.mean(db1, axis=0)
            self.w2 = self.w2 - learning_rate * np.mean(dw2, axis=0)
            self.b2 = self.b2 - learning_rate * np.mean(db2, axis=0)

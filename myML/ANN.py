import numpy as np

class ANN_Reg:
    def __init__(self, num_hidden_layers=(100, 100), eta=0.01, max_iter=1000, tol=0.0001):
        self.num_hlayers = num_hidden_layers
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol

    def __sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def __get_z(self, x, W):
        return np.matmul(x, W)

    def __get_error(self, y, y_predict):
        return np.sum((y - y_predict) ** 2) / len(y)
        
    def __backpropagation(self, X, y):
        m, n = X.shape
        n_out = y.shape[1] if len(y)==2 else 1

        layer_size = self.num_hlayers + (n_out,)
        num_layer = len(layer_size)

        W = []
        li_n = n
        for lj_n in layer_size:
            Wij = np.random.rand(li_n + 1, lj_n) * 0.05
            W.append(Wij)
            li_n = lj_n

        in_list      = [None] * num_layer
        z_list       = [None] * num_layer 
        out     = [None] * num_layer
        delta   = [None] * num_layer

        idx = np.arange(m)
        for _ in range(self.max_iter):
            np.random.shuffle(idx)
            X, y = X[idx], y[idx]

            for x, t in zip(X, y):
                out = x
                for i in range(num_layer):
                    in_ = np.ones(out.size + 1)
                    in_[1:] = out 
                    z = self.__get_z(in_, W[i])
                    if i != num_layer - 1:
                        out = self.__sigmoid(z)
                    else:
                        out = z
                    in_list[i], z_list[i], out[i] = in_, z, out


                delta[-1] = t - out
                for i in range(num_layer - 2, -1, -1):
                    out_i, W_j, delta_j = out[i], W[i+1], delta[i+1]
                    delta[i] = out_i * (1. - out_i) * np.matmul(W_j[1:], delta_j[:, None]).T[0]

                for i in range(num_layer):
                    in_i, delta_i = in_list[i], delta[i]
                    W[i] += in_i[:, None] * delta_i * self.eta

            y_pred = self.__forward(X, W)
            err = self.__get_error(y, y_pred)

            if err < self.tol:
                break

        return W

    def train(self, X, y):
        self.W = self.__backpropagation(X, y)

    def __forward(self, X, W):
        layer_n = len(W)

        out = X
        for i in range(layer_n):
            m, n = out.shape
            in_ = np.ones((m, n + 1))
            in_[:, 1:] = out
            z = self.__get_z(in_, W[i])
            if i != layer_n - 1:
                out = self.__sigmoid(z)
            else:
                out = z

        return out

    def predict(self, X):
        return self.__forward(X, self.W)

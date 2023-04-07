from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, NewType, Tuple
import numpy as np

DataSet = NewType('DataSet', List[Tuple[np.ndarray, int]])


class Net:
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden_layer_sizes: List[int],
    ) -> None:
        self._layer_size = [n_inputs, *hidden_layer_sizes, n_outputs] 
        self._n_layers = len(self._layer_size)
        # Note: weight matrix has dimensions (layer_size[n+1], layer_size[n])
        #       so we can obtain n+1 layer activations simply by sigmoid(np.matmul(weight[n],activ[n]) + bias[n])
        self._weights = [np.random.randn(n2, n1) 
                         for n1, n2 in zip(self._layer_size[:-1], self._layer_size[1:])]
        self._biases = [np.random.randn(n, 1) for n in self._layer_size[1:]]
        
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x) * (1 - self._sigmoid(x))
    
    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        activation = inputs
        for b, w in zip(self._biases, self._weights):
            activation = self._sigmoid(np.matmul(w, activation) + b)
        return activation
    
    def update_parameters(self, mini_batch: DataSet, learning_rate: float) -> None:
        grad_b = [np.zeros_like(b) for b in self._biases]
        grad_w = [np.zeros_like(w) for w in self._weights]
        
        for data, target in mini_batch:
            delta_grad_b, delta_grad_w = self._back_prop(data, target)
            grad_b = self._get_new_gradients(grad_b, delta_grad_b)
            grad_w = self._get_new_gradients(grad_w, delta_grad_w)

        self._weights = self._get_new_parameters(
            self._weights, grad_w, learning_rate, len(mini_batch))
        
        self._biases = self._get_new_parameters(
            self._biases, grad_b, learning_rate, len(mini_batch))
        
    def evaluate(self, test_set: DataSet) -> int:
        # Calculate number of correct answers
        results = [(np.argmax(self.forward_pass(data)), target) for data, target in test_set]
        return sum(int(pred == target) for pred, target in results)
    
    def _cost_derivative(self, output_activations, targets):
        # Cost for a data sample x is 0.5 * (activations(x) - target)**2
        return output_activations - targets
        
    @staticmethod
    def _get_new_gradients(grad_x: List[np.ndarray], delta_grad_x: List[np.ndarray]) -> List[np.ndarray]:
        return [x + delta_x for x, delta_x in zip(grad_x, delta_grad_x)]
    
    def _get_new_parameters(
        self,
        params: List[np.ndarray],
        grad_params: List[np.ndarray],
        learning_rate: float,
        sample_size: int
    ) -> List[np.ndarray]:
        return [p - learning_rate * grad_p / sample_size for p, grad_p in zip(params, grad_params)]
    
    def _back_prop(self, data, target):
        grad_b = [np.zeros_like(b) for b in self._biases]
        grad_w = [np.zeros_like(w) for w in self._weights]

        activations = [data]
        weighted_inputs = []
        for b, w in zip(self._biases, self._weights):
            weighted_inputs.append(np.matmul(w, activations[-1]) + b)
            activations.append(self._sigmoid(weighted_inputs[-1]))

        error = self._cost_derivative(activations[-1], target) * \
            self._sigmoid_derivative(weighted_inputs[-1])
            
        grad_b[-1] = error
        grad_w[-1] = np.matmul(error, activations[-2].transpose())

        # for layers L - 1 ... 2, but indexes are L - 2 ... 1
        for l in range(self._n_layers - 2, 0, -1):
            
            sig_der = self._sigmoid_derivative(weighted_inputs[l - 1])
            # weights, grad_b and grad_w lists have n_layers - 1 elements
            error = np.matmul(self._weights[l].transpose(), error) * sig_der
            grad_b[l - 1] = error
            grad_w[l - 1] = np.matmul(error, activations[l - 1].transpose())
 
        return (grad_b, grad_w)
    

class SGD:
    def __init__(
        self,
        train_set: DataSet,
        test_set: DataSet,
        mini_batch_size: int,
        epochs: int,
        learning_rate: float
    ) -> None:
        self._train_set = train_set
        self._test_set = test_set
        self._n_train = len(train_set)
        self._n_test = len(test_set)
        self._mini_batch_size = mini_batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate
        
    def train(self, net: Net):
        for epoch in range(self._epochs):
            mini_batches = self._shuffle_train_set_and_make_mini_batches()
            for mini_batch in mini_batches:
                net.update_parameters(mini_batch, self._learning_rate)
            self._log_performance(net, epoch)

    def _log_performance(self, net: Net, epoch: int) -> None:
        n_correct = net.evaluate(self._test_set)
        percentage = n_correct / self._n_test
        percentage_str = "{:.2%}".format(percentage)
        print(f"Epoch {epoch}/{self._epochs}: ACC {n_correct}/{self._n_test}, {percentage_str}")    
    
    def _shuffle_train_set_and_make_mini_batches(self) -> Iterable[DataSet]:
        np.random.shuffle(self._train_set)
        return [
            self._train_set[k: k + self._mini_batch_size] for k in range(0, self._n_train, self._mini_batch_size)
        ]

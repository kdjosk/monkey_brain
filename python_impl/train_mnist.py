from typing import NewType, Tuple
import numpy as np
from tensorflow import keras
from net import Net, DataSet, SGD


def rescale_grayscale_image_values(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image /= 255.0
    image -= image.mean()
    return image


def one_hot_encoding(bit, length) -> np.ndarray:
    enc = np.zeros((length, 1))
    enc[bit] = 1.0
    return enc


def load_mnist() -> Tuple[DataSet, DataSet]:
    (train_data, train_target), (test_data, test_target) = keras.datasets.mnist.load_data()
    
    train_data = [np.reshape(x, (784, 1)) for x in train_data]
    test_data = [np.reshape(x, (784, 1)) for x in test_data]
    
    train_data = [rescale_grayscale_image_values(x) for x in train_data]
    test_data = [rescale_grayscale_image_values(x) for x in test_data]
    
    train_set = [(x, one_hot_encoding(y, 10)) for x, y in zip(train_data, train_target)]
    test_set = [(x, y) for x, y in zip(test_data, test_target)]

    return train_set, test_set


def main():
    train_set, test_set = load_mnist()
    sgd = SGD(train_set, test_set, mini_batch_size=10, epochs=30, learning_rate=3.0)
    net = Net(n_inputs=784, n_outputs=10, hidden_layer_sizes=[30])
    sgd.train(net)
    

if __name__ == "__main__":
    main()
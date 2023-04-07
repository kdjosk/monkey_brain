use ndarray::{Array2, Array1, Array};
use ndarray_rand::RandomExt;
use rand::seq::SliceRandom;
use rand::thread_rng;
use ndarray_rand::rand_distr::StandardNormal;

#[derive(Debug)]
pub struct Net {
    n_layers: usize,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>
}

impl Net {
    pub fn new(
        layer_sizes: Vec<usize>,
    ) -> Net {
        let n_layers = layer_sizes.len();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        
        for layer in 1..n_layers {
            let b = Array::random(layer_sizes[layer], StandardNormal);
            let w = Array::random(
                (layer_sizes[layer], layer_sizes[layer - 1]),
                StandardNormal,
            );

            biases.push(b);
            weights.push(w);
        }

        Net {
            n_layers,
            weights,
            biases
        }
    }

    fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|a| 1.0 / (1.0 + (-a).exp()))
    }

    fn sigmoid_prime(x: &Array1<f64>) -> Array1<f64> {
        let sigx = Net::sigmoid(x);
        sigx.mapv(|a| 1.0 - a) * sigx 
    }

    pub fn forward_pass(&self, inputs: &Array1<f64>) -> Array1<f64> {
        let mut a = inputs.clone();
        for (b, w) in self.biases.iter().zip(&self.weights) {
            let z = w.dot(&a) + b;
            a = Net::sigmoid(&z);
        }
        a
    }

    pub fn update_parameters(
        &mut self, 
        mini_batch: &[(Array1<f64>, Array1<f64>)], 
        learn_rate: f64
    ) {
        let mut grad_b = Vec::new();
        let mut grad_w = Vec::new();
        for (b, w) in self.biases.iter().zip(&self.weights) {
            grad_b.push(Array1::<f64>::zeros(b.raw_dim()));
            grad_w.push(Array2::<f64>::zeros(w.raw_dim()));
        }

        for(x, y) in mini_batch.iter() {
            let (delta_grad_b, delta_grad_w) = self.back_prop(x, y);
            for (gb, dgb) in grad_b.iter_mut().zip(&delta_grad_b) {
                *gb += dgb;
            }
            for (gw, dgw) in grad_w.iter_mut().zip(&delta_grad_w) {
                *gw += dgw;
            }
        }

        let scaled_rate = learn_rate / mini_batch.len() as f64;
        for (w, gw) in self.weights.iter_mut().zip(&grad_w) {
            *w -= &(scaled_rate * gw);
        }
        for (b, gb) in self.biases.iter_mut().zip(&grad_b) {
            *b -= &(scaled_rate * gb);
        }
    }

    pub fn evaluate(&self, test_set: &[(Array1<f64>, usize)]) -> usize {
        // Calculate number of correct answers
        let mut correct = 0;
        for (x, y) in test_set.iter() {
            let out = Net::argmax(&self.forward_pass(x));
            if &out == y { correct += 1};
        }
        correct
    }

    fn argmax(arr: &Array1<f64>) -> usize {
        let mut max = f64::MIN;
        let mut i_max = 0;
        for (i, el) in arr.iter().enumerate() {
            if el > &max {
                i_max = i;
                max = *el;
            }
        }
        i_max
    }

    fn back_prop(
        &self, x: &Array1<f64>, y: &Array1<f64>
    ) -> (Vec<Array1<f64>>, Vec<Array2<f64>>) {
        (Vec::new(), Vec::new())
    }

}


pub struct SGD {
    train_set: Vec<(Array1<f64>, Array1<f64>)>,
    test_set: Vec<(Array1<f64>, usize)>,
    mini_batch_size: usize,
    epochs: usize,
    learn_rate: f64,
}

impl SGD {
    pub fn new(
        train_set: Vec<(Array1<f64>, Array1<f64>)>,
        test_set: Vec<(Array1<f64>, usize)>,
        mini_batch_size: usize,
        epochs: usize,
        learn_rate: f64,
    ) -> SGD {
        SGD {
            train_set,
            test_set,
            mini_batch_size,
            epochs,
            learn_rate,
        }
    }

    pub fn train(&self, net: &mut Net) {
        let mut rng = thread_rng();
        let mut train_set = self.train_set.clone();

        for epoch in 0..self.epochs {
            train_set.shuffle(&mut rng);
            for mini_batch in train_set.chunks(self.mini_batch_size) {
                net.update_parameters(mini_batch, self.learn_rate)
            }
        }
    }

    fn log_performance(self, net: Net, epoch: usize) {
        let n_correct = net.evaluate(&self.test_set);
    }
}


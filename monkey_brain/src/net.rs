
use ndarray::{Array2, Array, Axis, concatenate};
use ndarray_rand::RandomExt;
use rand::seq::SliceRandom;
use rand::thread_rng;
use ndarray_rand::rand_distr::StandardNormal;

#[derive(Debug)]
pub struct Net {
    n_layers: usize,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>
}

impl Net {
    pub fn new(
        layer_sizes: Vec<usize>,
    ) -> Net {
        let n_layers = layer_sizes.len();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        
        for layer in 1..n_layers {
            let b = Array::random((layer_sizes[layer], 1), StandardNormal);
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

    fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|a| 1.0 / (1.0 + (-a).exp()))
    }

    fn sigmoid_prime(x: &Array2<f64>) -> Array2<f64> {
        let sigx = Net::sigmoid(x);
        sigx.mapv(|a| 1.0 - a) * sigx 
    }

    fn cost_prime(output_activations: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        // partial derivatives vector of cost w.r.t. output activations
        return output_activations - targets
    }

    pub fn forward_pass(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut a = inputs.clone();
        for (b, w) in self.biases.iter().zip(&self.weights) {
            let z = w.dot(&a) + b;
            a = Net::sigmoid(&z);
        }
        a
    }

    pub fn update_parameters(
        &mut self, 
        mini_batch: &[(Array2<f64>, Array2<f64>)], 
        learn_rate: f64
    ) {
        let mut mb_iter = mini_batch.iter();
        let mb_first = mb_iter.next().unwrap();
        let mut x_mat = mb_first.0.clone();
        let mut y_mat = mb_first.1.clone();

        for (x, y) in mb_iter {
            x_mat = concatenate![Axis(1), x_mat, *x];
            y_mat = concatenate![Axis(1), y_mat, *y];
        }

        let (grad_b, grad_w) = self.backprop(&x_mat, &y_mat);

        let scaled_rate = learn_rate / mini_batch.len() as f64;
        for (w, gw) in self.weights.iter_mut().zip(&grad_w) {
            *w -= &(scaled_rate * gw);
        }
        for (b, gb) in self.biases.iter_mut().zip(&grad_b) {
            *b -= &(scaled_rate * gb);
        }
    }

    pub fn evaluate(&self, test_set: &[(Array2<f64>, usize)]) -> usize {
        // Calculate number of correct answers
        let mut correct = 0;
        for (x, y) in test_set.iter() {
            let out = Net::argmax(&self.forward_pass(x));
            if &out == y { correct += 1};
        }
        correct
    }

    fn argmax(arr: &Array2<f64>) -> usize {
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

    fn backprop(
        &self, x: &Array2<f64>, y: &Array2<f64>
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        // Return a tuple (grad_b, grad_w) representing the
        // gradient for the cost function. 
        let mut grad_b = Vec::new();
        let mut grad_w = Vec::new();

        for (b, w) in self.biases.iter().zip(&self.weights) {
            grad_b.push(Array2::<f64>::zeros(b.raw_dim()));
            grad_w.push(Array2::<f64>::zeros(w.raw_dim()));
        }

        let mut activations = vec![x.clone()];
        let mut weighted_inputs = Vec::new();

        for (b, w) in self.biases.iter().zip(&self.weights) {
            let previous_activation = activations.last().unwrap();
            let z = w.dot(previous_activation) + b;
            activations.push(Net::sigmoid(&z));
            weighted_inputs.push(z);
        }

        // 1st equation for error in the output layer
        let error = Net::cost_prime(
            &activations.pop().unwrap(), y
        ) * Net::sigmoid_prime(weighted_inputs.last().unwrap());

        let act = activations
            .pop()
            .unwrap();
        
        *grad_w.last_mut().unwrap() = error.dot(&act.t());
        *grad_b.last_mut().unwrap() = error.sum_axis(Axis(1)).insert_axis(Axis(1));
        
        for l in (1 ..= self.n_layers - 2).rev() {
            // segond equation for error in terms of the error
            // in the next layer
            // l is the index of the next layer
            // for 3 layers, the layer idxs are 0, 1, 2
            // the weights w_1 and w_2 have indexes w[0] and w[1]
            let error = self.weights[l].t().dot(&error)
                        * Net::sigmoid_prime(&weighted_inputs[l - 1]);

            grad_b[l - 1] = error.sum_axis(Axis(1)).insert_axis(Axis(1));
      
            let act = activations
                .pop()
                .unwrap();

            grad_w[l - 1] = error.dot(&act.t());
        }
        
        (grad_b, grad_w)
    }
}


pub struct SGD {
    train_set: Vec<(Array2<f64>, Array2<f64>)>,
    test_set: Vec<(Array2<f64>, usize)>,
    mini_batch_size: usize,
    epochs: usize,
    learn_rate: f64,
}

impl SGD {
    pub fn new(
        train_set: Vec<(Array2<f64>, Array2<f64>)>,
        test_set: Vec<(Array2<f64>, usize)>,
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
            self.log_performance(net, epoch);
        }
    }

    fn log_performance(&self, net: &Net, epoch: usize) {
        let n_correct = net.evaluate(&self.test_set);
        let n_samples = self.test_set.len();
        let percentage = n_correct as f64 / n_samples as f64;
        println!("Epoch {} / {}: ACC {} / {} = {}", epoch, self.epochs, n_correct, n_samples, percentage);
    }
}

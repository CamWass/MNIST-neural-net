#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::time::Instant;

use rand::Rng;
use rand_distr::{DistIter, StandardNormal};

/// An `m` by `n` matrix i.e. `m` rows and `n` columns.
type Matrix<T, const M: usize, const N: usize> = [[T; N]; M];

struct Params {
    W1: Matrix<f64, 128, 784>,
    W2: Matrix<f64, 64, 128>,
    W3: Matrix<f64, 10, 64>,

    A0: [f64; 784],
    A1: [f64; 128],
    A2: [f64; 64],
    A3: [f64; 10],

    Z1: [f64; 128],
    Z2: [f64; 64],
    Z3: [f64; 10],
}

impl Default for Params {
    fn default() -> Self {
        Self {
            W1: [[0.0; 784]; 128],
            W2: [[0.0; 128]; 64],
            W3: [[0.0; 64]; 10],

            A0: [0.0; 784],
            A1: [0.0; 128],
            A2: [0.0; 64],
            A3: [0.0; 10],

            Z1: [0.0; 128],
            Z2: [0.0; 64],
            Z3: [0.0; 10],
        }
    }
}

struct WeightDeltas {
    W1: Matrix<f64, 128, 784>,
    W2: Matrix<f64, 64, 128>,
    W3: Matrix<f64, 10, 64>,
}

impl Default for WeightDeltas {
    fn default() -> Self {
        Self {
            W1: [[0.0; 784]; 128],
            W2: [[0.0; 128]; 64],
            W3: [[0.0; 64]; 10],
        }
    }
}

pub struct NeuralNet {
    iterations: usize,
    learn_rate: f64,
    params: Box<Params>,
}

impl NeuralNet {
    pub fn new() -> Self {
        let mut params = Box::new(Params::default());

        let mut rng: DistIter<_, _, f64> = rand::thread_rng().sample_iter(StandardNormal);

        // let mut rng = rand::thread_rng();

        let input_layer = 784;
        let hidden_1 = 128;
        let hidden_2 = 64;
        let output_layer = 10;

        for row in &mut params.W1 {
            for element in row {
                *element = rng.next().unwrap() * (1.0 / hidden_1 as f64).sqrt();
            }
        }
        for row in &mut params.W2 {
            for element in row {
                *element = rng.next().unwrap() * (1.0 / hidden_2 as f64).sqrt();
            }
        }
        for row in &mut params.W3 {
            for element in row {
                *element = rng.next().unwrap() * (1.0 / output_layer as f64).sqrt();
            }
        }

        Self {
            // iterations: 10,
            iterations: 2,
            learn_rate: 0.001,
            // learn_rate: 0.1,
            params,
        }
    }

    fn forward_pass(&mut self, image: &[f64; 784]) {
        // input layer activations becomes sample
        self.params.A0.copy_from_slice(image);

        assert!(self.params.W1.iter().all(|w| w.iter().all(|&w| w <= 1.0)));
        assert!(self.params.A0.iter().all(|&w| w <= 1.0));

        // input layer to hidden layer 1
        self.params.Z1 = matrix_multiply(&self.params.W1, &self.params.A0);
        self.params.A1 = sigmoid(&self.params.Z1, false);

        // hidden layer 1 to hidden layer 2
        self.params.Z2 = matrix_multiply(&self.params.W2, &self.params.A1);
        self.params.A2 = sigmoid(&self.params.Z2, false);

        // hidden layer 2 to output layer
        self.params.Z3 = matrix_multiply(&self.params.W3, &self.params.A2);
        self.params.A3 = softmax(&self.params.Z3, false);
    }

    fn backward_pass(&mut self, target: &[f64; 10]) -> Box<WeightDeltas> {
        // This is the backpropagation algorithm, for calculating the updates
        // of the neural network's parameters.

        let output = &self.params.A3;

        let mut change_w = Box::new(WeightDeltas::default());

        // Calculate W3 update

        let mut error = softmax(&self.params.Z3, true);
        for ((error, output), target) in error.iter_mut().zip(output).zip(target) {
            *error = 2.0 * (output - target) / 10.0 * *error
        }
        change_w.W3 = outer_product(&error, &self.params.A2);

        // Calculate W2 update

        let Z2_sigmoid = sigmoid(&self.params.Z2, true);
        let W3_transpose = transpose(&self.params.W3);
        let mut error = matrix_multiply(&W3_transpose, &error);
        for (a, b) in error.iter_mut().zip(Z2_sigmoid) {
            *a = *a * b;
        }
        change_w.W2 = outer_product(&error, &self.params.A1);

        // Calculate W1 update

        let Z1_sigmoid = sigmoid(&self.params.Z1, true);
        let W2_transpose = transpose(&self.params.W2);
        let mut error = matrix_multiply(&W2_transpose, &error);
        for (a, b) in error.iter_mut().zip(Z1_sigmoid) {
            *a = *a * b;
        }
        change_w.W1 = outer_product(&error, &self.params.A0);

        change_w
    }

    fn update_network_parameters(&mut self, changes_to_w: &WeightDeltas) {
        // Update network parameters according to update rule from
        // Stochastic Gradient Descent.
        //
        // θ = θ - η * ∇J(x, y),
        //     theta θ:            a network parameter (e.g. a weight w)
        //     eta η:              the learning rate
        //     gradient ∇J(x, y):  the gradient of the objective function,
        //                         i.e. the change for a specific theta θ

        fn update_params<const M: usize, const N: usize>(
            weights: &mut Matrix<f64, M, N>,
            deltas: &Matrix<f64, M, N>,
            learn_rate: f64,
        ) {
            for (row, row_deltas) in weights.iter_mut().zip(deltas) {
                for (w, delta) in row.iter_mut().zip(row_deltas) {
                    *w -= learn_rate * delta;
                }
            }
        }

        update_params(&mut self.params.W1, &changes_to_w.W1, self.learn_rate);
        update_params(&mut self.params.W2, &changes_to_w.W2, self.learn_rate);
        update_params(&mut self.params.W3, &changes_to_w.W3, self.learn_rate);
    }

    fn compute_accuracy(&mut self, images: &[[f64; 784]], labels: &[[f64; 10]]) -> f64 {
        // This function does a forward pass of x, then checks if the indices
        // of the maximum value in the output equals the indices in the label
        // y. Then it sums over each prediction and calculates the accuracy.
        let mut correct_predictions = 0;

        for (image, label) in images.iter().zip(labels) {
            self.forward_pass(image);
            let pred = self
                .params
                .A3
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index);
            let target = label
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index);
            if pred == target {
                correct_predictions += 1;
            }
        }

        correct_predictions as f64 / images.len() as f64
    }

    pub fn train(
        &mut self,
        train_images: &[[f64; 784]],
        train_labels: &[[f64; 10]],
        test_images: &[[f64; 784]],
        test_labels: &[[f64; 10]],
    ) {
        let start_time = Instant::now();
        for iteration in 0..self.iterations {
            // TODO: remove take()
            for (x, y) in train_images
                .iter()
                .take(10_000)
                .zip(train_labels.iter().take(10_000))
            {
                // dbg!(&self.params.W3);
                // let old = self.params.W3.clone();
                // let output = self.forward_pass(x);
                // let changes_to_w = self.backward_pass(y, output);
                // self.update_network_parameters(changes_to_w);
                self.forward_pass(x);
                let changes_to_w = self.backward_pass(y);
                self.update_network_parameters(&changes_to_w);
                // dbg!(&self.params.W3);
                // println!("params changed: {}", old != self.params.W3);
            }

            let accuracy = self.compute_accuracy(test_images, test_labels);
            println!(
                "Iteration: {}, Time elapsed: {}s, Accuracy: {}%",
                iteration,
                start_time.elapsed().as_secs(),
                accuracy * 100.0
            );
        }
        // self.forward_pass(&train_images[0]);
        // let changes_to_w = self.backward_pass(&train_labels[0]);
        // self.update_network_parameters(changes_to_w);
    }
}

fn outer_product<const M: usize, const N: usize>(a: &[f64; M], b: &[f64; N]) -> Matrix<f64, M, N> {
    let mut result = [[0.0; N]; M];

    for row in 0..M {
        for col in 0..N {
            result[row][col] = a[row] * b[col]
        }
    }

    result
}

fn transpose<const M: usize, const N: usize>(matrix: &Matrix<f64, M, N>) -> Matrix<f64, N, M> {
    let mut transpose = [[0.0; M]; N];

    for row in 0..N {
        for col in 0..M {
            transpose[row][col] = matrix[col][row];
        }
    }

    transpose
}

fn matrix_multiply<const M: usize, const N: usize>(
    a: &Matrix<f64, M, N>,
    b: &[f64; N],
) -> [f64; M] {
    let mut result = [0.0; M];

    for (i, row) in a.iter().enumerate() {
        result[i] = row.iter().zip(b).map(|(a, b)| a * b).sum();
    }

    result
}

fn sigmoid<const N: usize>(x: &[f64; N], derivative: bool) -> [f64; N] {
    let mut result = [0.0; N];
    if derivative {
        for (i, &x) in x.iter().enumerate() {
            result[i] = ((-x).exp()) / (((-x).exp() + 1.0).powf(2.0));
        }
    } else {
        for (i, &x) in x.iter().enumerate() {
            result[i] = 1.0 / (1.0 + (-x).exp());
        }
    }
    result
}

fn softmax<const N: usize>(x: &[f64; N], derivative: bool) -> [f64; N] {
    // Numerically stable with large exponentials

    let mut result = [0.0; N];

    let max = x.iter().copied().reduce(f64::max).unwrap();
    let exp_sum: f64 = x.iter().map(|&x| (x - max).exp()).sum();
    if derivative {
        for (i, &x) in x.iter().enumerate() {
            result[i] = (x - max).exp() / exp_sum * (1.0 - (x - max).exp() / exp_sum)
        }
    } else {
        for (i, &x) in x.iter().enumerate() {
            result[i] = (x - max).exp() / exp_sum;
        }
    }
    result
}

// pub struct NeuralNetBuilder {}

// impl NeuralNetBuilder {
//     pub fn new() -> Self {
//         todo!();
//     }

//     pub fn input_layer(self) -> Self {
//         todo!();
//     }
//     pub fn hidden_layer(self) -> Self {
//         todo!();
//     }
//     pub fn output_layer(self) -> Self {
//         todo!();
//     }

//     pub fn build(self) -> NeuralNet {
//         todo!();
//     }
// }

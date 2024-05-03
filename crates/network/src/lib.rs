mod util;

use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::{DistIter, StandardNormal};
use serde::{Deserialize, Serialize};

use util::*;

struct Params {
    a1: [f32; 128],
    a2: [f32; 64],
    a3: [f32; 10],

    z1: [f32; 128],
    z2: [f32; 64],
    z3: [f32; 10],
}

impl Default for Params {
    fn default() -> Self {
        Self {
            a1: [0.0; 128],
            a2: [0.0; 64],
            a3: [0.0; 10],

            z1: [0.0; 128],
            z2: [0.0; 64],
            z3: [0.0; 10],
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
struct Weights {
    w1: Matrix<f32, 128, 784, 100_352>,
    w2: Matrix<f32, 64, 128, 8192>,
    w3: Matrix<f32, 10, 64, 640>,
}

#[derive(Serialize, Deserialize)]
pub struct State {
    activation_fn: ActivationFunction,
    weights: Weights,
}

const LEARN_RATE: f32 = 0.001;
const MOMENTUM: f32 = 0.9;

pub struct NeuralNet {
    activation_fn: ActivationFunction,
    weights: Weights,
    params: Box<Params>,
    weight_deltas: Weights,
    prev_weight_deltas: Weights,
}

impl NeuralNet {
    pub fn new() -> Self {
        Self::init(rand::thread_rng().gen())
    }

    pub fn new_for_bench(rng_seed: u64) -> Self {
        Self::init(rng_seed)
    }

    fn init(rnd_seed: u64) -> Self {
        let mut weights = Weights::default();

        let mut rng: DistIter<_, _, f32> =
            SmallRng::seed_from_u64(rnd_seed).sample_iter(StandardNormal);

        let hidden_1 = 128;
        let hidden_2 = 64;
        let output_layer = 10;

        for element in weights.w1.iter_mut() {
            *element = rng.next().unwrap() * (1.0 / hidden_1 as f32).sqrt();
        }
        for element in weights.w2.iter_mut() {
            *element = rng.next().unwrap() * (1.0 / hidden_2 as f32).sqrt();
        }
        for element in weights.w3.iter_mut() {
            *element = rng.next().unwrap() * (1.0 / output_layer as f32).sqrt();
        }

        Self {
            activation_fn: ActivationFunction::RectifiedLinearUnit,
            weights,
            params: Box::new(Params::default()),
            weight_deltas: Weights::default(),
            prev_weight_deltas: Weights::default(),
        }
    }

    fn forward_pass(&mut self, image: &[f32; 784]) {
        // input layer to hidden layer 1
        self.params.z1 = self.weights.w1.multiply_by(&image);
        self.params.a1 = self.call_activation_fn(&self.params.z1, false);

        // hidden layer 1 to hidden layer 2
        self.params.z2 = self.weights.w2.multiply_by(&self.params.a1);
        self.params.a2 = self.call_activation_fn(&self.params.z2, false);

        // hidden layer 2 to output layer
        self.params.z3 = self.weights.w3.multiply_by(&self.params.a2);
        self.params.a3 = softmax(&self.params.z3, false);
    }

    fn backward_pass(&mut self, image: &[f32; 784], target: &[f32; 10]) {
        // This is the backpropagation algorithm, for calculating the updates
        // of the neural network's parameters.

        let output = &self.params.a3;

        // Calculate W3 update

        let mut error = softmax(&self.params.z3, true);
        for ((error, output), target) in error.iter_mut().zip(output).zip(target) {
            *error = 2.0 * (output - target) / 10.0 * *error
        }
        outer_product(&mut self.weight_deltas.w3, &error, &self.params.a2);

        // Calculate W2 update

        let z2_sigmoid = self.call_activation_fn(&self.params.z2, true);
        let mut error = self.weights.w3.transpose().multiply_by(&error);
        for (a, b) in error.iter_mut().zip(z2_sigmoid) {
            *a = *a * b;
        }
        outer_product(&mut self.weight_deltas.w2, &error, &self.params.a1);

        // Calculate W1 update

        let z1_sigmoid = self.call_activation_fn(&self.params.z1, true);
        let mut error = self.weights.w2.transpose().multiply_by(&error);
        for (a, b) in error.iter_mut().zip(z1_sigmoid) {
            *a = *a * b;
        }
        outer_product(&mut self.weight_deltas.w1, &error, image);
    }

    fn call_activation_fn<const N: usize>(&self, x: &[f32; N], derivative: bool) -> [f32; N] {
        match self.activation_fn {
            ActivationFunction::Sigmoid => sigmoid(x, derivative),
            ActivationFunction::RectifiedLinearUnit => rectified_linear_unit(x, derivative),
        }
    }

    fn update_network_parameters(&mut self) {
        // Update network parameters according to update rule from
        // Stochastic Gradient Descent.
        //
        // θ = θ - η * ∇J(x, y),
        //     theta θ:            a network parameter (e.g. a weight w)
        //     eta η:              the learning rate
        //     gradient ∇J(x, y):  the gradient of the objective function,
        //                         i.e. the change for a specific theta θ

        fn update_params<const M: usize, const N: usize, const S: usize>(
            weights: &mut Matrix<f32, M, N, S>,
            deltas: &Matrix<f32, M, N, S>,
            prev_deltas: &mut Matrix<f32, M, N, S>,
        ) {
            for ((w, delta), prev_delta) in weights
                .iter_mut()
                .zip(deltas.iter())
                .zip(prev_deltas.iter_mut())
            {
                let new_delta = LEARN_RATE * delta + MOMENTUM * *prev_delta;
                *w -= new_delta;
                *prev_delta = new_delta;
            }
        }

        update_params(
            &mut self.weights.w1,
            &self.weight_deltas.w1,
            &mut self.prev_weight_deltas.w1,
        );
        update_params(
            &mut self.weights.w2,
            &self.weight_deltas.w2,
            &mut self.prev_weight_deltas.w2,
        );
        update_params(
            &mut self.weights.w3,
            &self.weight_deltas.w3,
            &mut self.prev_weight_deltas.w3,
        );
    }

    pub fn compute_accuracy(&mut self, images: &[[f32; 784]], labels: &[[f32; 10]]) -> Accuracy {
        // This function does a forward pass of x, then checks if the indices
        // of the maximum value in the output equals the indices in the label
        // y. Then it sums over each prediction and calculates the accuracy.
        let mut correct_predictions = 0;

        let mut per_element = [Guesses::default(); 10];

        for (image, label) in images.iter().zip(labels) {
            self.forward_pass(image);
            let pred = self
                .params
                .a3
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap();
            let target = label
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap();

            per_element[target].total += 1;
            if pred == target {
                correct_predictions += 1;
                per_element[target].correct += 1;
            }
        }

        Accuracy {
            overall: Guesses::new(images.len(), correct_predictions),
            per_element,
        }
    }

    /// Only pub for benchmarks.
    pub fn train_one(&mut self, train_image: &[f32; 784], train_label: &[f32; 10]) {
        self.forward_pass(train_image);
        self.backward_pass(train_image, train_label);
        self.update_network_parameters();
    }

    pub fn train(&mut self, train_images: &[[f32; 784]], train_labels: &[[f32; 10]]) {
        for (x, y) in train_images.iter().zip(train_labels) {
            self.train_one(x, y);
        }
    }

    pub fn classify(&mut self, image: &[f32; 784]) -> &[f32; 10] {
        self.forward_pass(image);
        &self.params.a3
    }

    pub fn restore_from_state(state: State) -> Self {
        Self {
            activation_fn: state.activation_fn,
            params: Box::new(Params::default()),
            weights: state.weights,
            weight_deltas: Weights::default(),
            prev_weight_deltas: Weights::default(),
        }
    }

    pub fn into_state(self) -> State {
        State {
            activation_fn: self.activation_fn,
            weights: self.weights,
        }
    }

    pub fn set_activation_function(&mut self, activation_fn: ActivationFunction) {
        self.activation_fn = activation_fn;
    }
}

#[derive(Default, Copy, Clone)]
pub struct Guesses {
    pub total: usize,
    pub correct: usize,
}

impl Guesses {
    fn new(total: usize, correct: usize) -> Self {
        Self { total, correct }
    }
    pub fn percent_correct(&self) -> f32 {
        self.correct as f32 / self.total as f32 * 100.0
    }
}

#[derive(Default)]
pub struct Accuracy {
    /// Percent of correct guesses.
    pub overall: Guesses,
    /// Percent of correct guesses vs total occurances for each label type.
    pub per_element: [Guesses; 10],
}

fn outer_product<const M: usize, const N: usize, const S: usize>(
    output: &mut Matrix<f32, M, N, S>,
    a: &[f32; M],
    b: &[f32; N],
) {
    for row in 0..M {
        for col in 0..N {
            output.inner[row * N + col] = a[row] * b[col]
        }
    }
}

fn sigmoid<const N: usize>(x: &[f32; N], derivative: bool) -> [f32; N] {
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

fn softmax<const N: usize>(x: &[f32; N], derivative: bool) -> [f32; N] {
    // Numerically stable with large exponentials

    let mut result = [0.0; N];

    let max = x.iter().copied().reduce(f32::max).unwrap();
    let exp_sum: f32 = x.iter().map(|&x| (x - max).exp()).sum();
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

/// ReLU(x) = max(0, x)
fn rectified_linear_unit<const N: usize>(x: &[f32; N], derivative: bool) -> [f32; N] {
    let mut result = [0.0; N];
    if derivative {
        for i in 0..N {
            if x[i] > 0.0 {
                result[i] = 1.0;
            }
        }
    } else {
        for i in 0..N {
            if x[i] > 0.0 {
                result[i] = x[i];
            }
        }
    }

    result
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    RectifiedLinearUnit,
}

mod util;

use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::{DistIter, StandardNormal};
use serde::{Deserialize, Serialize};

use util::*;

#[derive(Serialize, Deserialize)]
pub struct Params {
    w1: Matrix<f32, 128, 784, 100_352>,
    w2: Matrix<f32, 64, 128, 8192>,
    w3: Matrix<f32, 10, 64, 640>,

    #[serde(with = "serde_arrays")]
    a0: [f32; 784],
    #[serde(with = "serde_arrays")]
    a1: [f32; 128],
    #[serde(with = "serde_arrays")]
    a2: [f32; 64],
    #[serde(with = "serde_arrays")]
    a3: [f32; 10],

    #[serde(with = "serde_arrays")]
    z1: [f32; 128],
    #[serde(with = "serde_arrays")]
    z2: [f32; 64],
    #[serde(with = "serde_arrays")]
    z3: [f32; 10],
}

impl Default for Params {
    fn default() -> Self {
        Self {
            w1: Matrix::default(),
            w2: Matrix::default(),
            w3: Matrix::default(),

            a0: [0.0; 784],
            a1: [0.0; 128],
            a2: [0.0; 64],
            a3: [0.0; 10],

            z1: [0.0; 128],
            z2: [0.0; 64],
            z3: [0.0; 10],
        }
    }
}

#[derive(Default)]
struct WeightDeltas {
    w1: Matrix<f32, 128, 784, 100_352>,
    w2: Matrix<f32, 64, 128, 8192>,
    w3: Matrix<f32, 10, 64, 640>,
}

const LEARN_RATE: f32 = 0.001;

pub struct NeuralNet {
    params: Box<Params>,
}

impl NeuralNet {
    pub fn new() -> Self {
        Self::init(rand::thread_rng().gen())
    }

    pub fn new_for_bench(rng_seed: u64) -> Self {
        Self::init(rng_seed)
    }

    fn init(rnd_seed: u64) -> Self {
        let mut params = Box::new(Params::default());

        let mut rng: DistIter<_, _, f32> =
            SmallRng::seed_from_u64(rnd_seed).sample_iter(StandardNormal);

        let hidden_1 = 128;
        let hidden_2 = 64;
        let output_layer = 10;

        for element in params.w1.iter_mut() {
            *element = rng.next().unwrap() * (1.0 / hidden_1 as f32).sqrt();
        }
        for element in params.w2.iter_mut() {
            *element = rng.next().unwrap() * (1.0 / hidden_2 as f32).sqrt();
        }
        for element in params.w3.iter_mut() {
            *element = rng.next().unwrap() * (1.0 / output_layer as f32).sqrt();
        }

        Self { params }
    }

    fn forward_pass(&mut self, image: &[f32; 784]) {
        // input layer activations becomes sample
        self.params.a0.copy_from_slice(image);

        assert!(self.params.w1.iter().all(|&w| w <= 1.0));
        assert!(self.params.a0.iter().all(|&w| w <= 1.0));

        // input layer to hidden layer 1
        self.params.z1 = matrix_multiply(&self.params.w1, &self.params.a0);
        self.params.a1 = sigmoid(&self.params.z1, false);

        // hidden layer 1 to hidden layer 2
        self.params.z2 = matrix_multiply(&self.params.w2, &self.params.a1);
        self.params.a2 = sigmoid(&self.params.z2, false);

        // hidden layer 2 to output layer
        self.params.z3 = matrix_multiply(&self.params.w3, &self.params.a2);
        self.params.a3 = softmax(&self.params.z3, false);
    }

    fn backward_pass(&mut self, target: &[f32; 10]) -> Box<WeightDeltas> {
        // This is the backpropagation algorithm, for calculating the updates
        // of the neural network's parameters.

        let output = &self.params.a3;

        let mut change_w = Box::new(WeightDeltas::default());

        // Calculate W3 update

        let mut error = softmax(&self.params.z3, true);
        for ((error, output), target) in error.iter_mut().zip(output).zip(target) {
            *error = 2.0 * (output - target) / 10.0 * *error
        }
        change_w.w3 = outer_product(&error, &self.params.a2);

        // Calculate W2 update

        let z2_sigmoid = sigmoid(&self.params.z2, true);
        let w3_transpose = self.params.w3.transpose();
        let mut error = matrix_multiply(&w3_transpose, &error);
        for (a, b) in error.iter_mut().zip(z2_sigmoid) {
            *a = *a * b;
        }
        change_w.w2 = outer_product(&error, &self.params.a1);

        // Calculate W1 update

        let z1_sigmoid = sigmoid(&self.params.z1, true);
        let w2_transpose = self.params.w2.transpose();
        let mut error = matrix_multiply(&w2_transpose, &error);
        for (a, b) in error.iter_mut().zip(z1_sigmoid) {
            *a = *a * b;
        }
        change_w.w1 = outer_product(&error, &self.params.a0);

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

        fn update_params<const M: usize, const N: usize, const S: usize>(
            weights: &mut Matrix<f32, M, N, S>,
            deltas: &Matrix<f32, M, N, S>,
        ) {
            for (w, delta) in weights.iter_mut().zip(deltas.iter()) {
                *w -= LEARN_RATE * delta;
            }
        }

        update_params(&mut self.params.w1, &changes_to_w.w1);
        update_params(&mut self.params.w2, &changes_to_w.w2);
        update_params(&mut self.params.w3, &changes_to_w.w3);
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
        let changes_to_w = self.backward_pass(train_label);
        self.update_network_parameters(&changes_to_w);
    }

    pub fn train(&mut self, train_images: &[[f32; 784]], train_labels: &[[f32; 10]]) {
        for (x, y) in train_images.iter().zip(train_labels) {
            self.train_one(x, y);
        }
    }

    pub fn classify(&mut self, image: &[f32; 784]) -> u8 {
        self.forward_pass(image);
        let prediction = self
            .params
            .a3
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap() as u8;
        prediction
    }

    pub fn from_params(params: Box<Params>) -> Self {
        Self { params }
    }

    pub fn get_params(self) -> Box<Params> {
        self.params
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
    a: &[f32; M],
    b: &[f32; N],
) -> Matrix<f32, M, N, S> {
    let mut result = Matrix::default();

    for row in 0..M {
        for col in 0..N {
            result.inner[row * N + col] = a[row] * b[col]
        }
    }

    result
}

fn matrix_multiply<const M: usize, const N: usize, const S: usize>(
    a: &Matrix<f32, M, N, S>,
    b: &[f32; N],
) -> [f32; M] {
    let mut result = [0.0; M];

    for (i, row) in a.rows().enumerate() {
        result[i] = row.iter().zip(b).map(|(a, b)| a * b).sum();
    }

    result
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

use std::time::Instant;

use rand::Rng;

use rand_distr::{DistIter, StandardNormal};

type Matrix<T> = Vec<Vec<T>>;

#[derive(Default, Clone, PartialEq)]
struct Params {
    W1: Matrix<f64>,
    W2: Matrix<f64>,
    W3: Matrix<f64>,

    A0: Vec<f64>,
    A1: Vec<f64>,
    A2: Vec<f64>,
    A3: Vec<f64>,

    Z1: Vec<f64>,
    Z2: Vec<f64>,
    Z3: Vec<f64>,
}

#[derive(Default)]
struct WeightDeltas {
    W1: Matrix<f64>,
    W2: Matrix<f64>,
    W3: Matrix<f64>,
}

pub struct NeuralNet {
    epochs: usize,
    learn_rate: f64,
    params: Params,
}

impl NeuralNet {
    pub fn new() -> Self {
        let mut params = Params::default();

        let mut rng: DistIter<_, _, f64> = rand::thread_rng().sample_iter(StandardNormal);

        // let mut rng = rand::thread_rng();

        let input_layer = 784;
        let hidden_1 = 128;
        let hidden_2 = 64;
        let output_layer = 10;

        params.W1 = vec![vec![0.0; input_layer]; hidden_1];
        for row in &mut params.W1 {
            for element in row {
                *element = rng.next().unwrap() * (1.0 / hidden_1 as f64).sqrt();
            }
        }
        params.W2 = vec![vec![0.0; hidden_1]; hidden_2];
        for row in &mut params.W2 {
            for element in row {
                *element = rng.next().unwrap() * (1.0 / hidden_2 as f64).sqrt();
            }
        }
        params.W3 = vec![vec![0.0; hidden_2]; output_layer];
        for row in &mut params.W3 {
            for element in row {
                *element = rng.next().unwrap() * (1.0 / output_layer as f64).sqrt();
            }
        }

        Self {
            // epochs: 10,
            epochs: 2,
            learn_rate: 0.001,
            // learn_rate: 0.1,
            params,
        }
    }

    fn forward_pass(&mut self, image: &[f64; 784]) {
        // input layer activations becomes sample
        self.params.A0.clear();
        self.params.A0.extend_from_slice(image);

        assert!(self.params.W1.iter().all(|w| w.iter().all(|&w| w <= 1.0)));
        assert!(self.params.A0.iter().all(|&w| w <= 1.0));

        // input layer to hidden layer 1
        self.params.Z1 = matrix_multiply(&self.params.W1, &self.params.A0);
        self.params.A1 = sigmoid(&self.params.Z1, false);

        // dbg!(&self.params.W1);
        // dbg!(&self.params.A0);
        // dbg!(&self.params.Z1);
        // dbg!(&self.params.A1);

        // hidden layer 1 to hidden layer 2
        self.params.Z2 = matrix_multiply(&self.params.W2, &self.params.A1);
        self.params.A2 = sigmoid(&self.params.Z2, false);

        // hidden layer 2 to output layer
        self.params.Z3 = matrix_multiply(&self.params.W3, &self.params.A2);
        self.params.A3 = softmax(&self.params.Z3, false);

        // println!("params = {{");
        // println!(
        //     "  W1: {:?}",
        //     (self.params.W1.len(), self.params.W1[0].len())
        // );
        // println!(
        //     "  W2: {:?}",
        //     (self.params.W2.len(), self.params.W2[0].len())
        // );
        // println!(
        //     "  W3: {:?}",
        //     (self.params.W3.len(), self.params.W3[0].len())
        // );
        // println!("");
        // println!("  A0: {}", (self.params.A0.len()));
        // println!("  A1: {}", (self.params.A1.len()));
        // println!("  A2: {}", (self.params.A2.len()));
        // println!("  A3: {}", (self.params.A3.len()));
        // println!("");
        // println!("  Z1: {}", (self.params.Z1.len()));
        // println!("  Z2: {}", (self.params.Z2.len()));
        // println!("  Z3: {}", (self.params.Z3.len()));
        // println!("");
        // println!("}}");

        // self.params.A3
    }

    fn backward_pass(&mut self, target: &[f64; 10]) -> WeightDeltas {
        // This is the backpropagation algorithm, for calculating the updates
        // of the neural network's parameters.

        let output = &self.params.A3;

        let mut change_w = WeightDeltas::default();

        // Calculate W3 update

        let soft_max = softmax(&self.params.Z3, true);
        let error = output
            .iter()
            .zip(target)
            .zip(soft_max)
            .map(|((o, t), s)| 2.0 * (o - t) / 10.0 * s)
            .collect::<Vec<_>>();
        change_w.W3 = outer_product(&error, &self.params.A2);

        assert_eq!(change_w.W3.len(), self.params.W3.len());
        assert_eq!(change_w.W3[0].len(), self.params.W3[0].len());

        // Calculate W2 update

        let Z2_sigmoid = sigmoid(&self.params.Z2, true);
        let W3_transpose = transpose(&self.params.W3);
        let error = matrix_multiply(&W3_transpose, &error)
            .into_iter()
            .zip(Z2_sigmoid)
            .map(|(a, b)| a * b)
            .collect();
        change_w.W2 = outer_product(&error, &self.params.A1);

        assert_eq!(change_w.W2.len(), self.params.W2.len());
        assert_eq!(change_w.W2[0].len(), self.params.W2[0].len());

        // Calculate W1 update

        let Z1_sigmoid = sigmoid(&self.params.Z1, true);
        let W2_transpose = transpose(&self.params.W2);
        let error = matrix_multiply(&W2_transpose, &error)
            .into_iter()
            .zip(Z1_sigmoid)
            .map(|(a, b)| a * b)
            .collect();
        change_w.W1 = outer_product(&error, &self.params.A0);

        assert_eq!(change_w.W1.len(), self.params.W1.len());
        assert_eq!(change_w.W1[0].len(), self.params.W1[0].len());

        change_w
    }

    fn update_network_parameters(&mut self, changes_to_w: WeightDeltas) {
        // Update network parameters according to update rule from
        // Stochastic Gradient Descent.
        //
        // θ = θ - η * ∇J(x, y),
        //     theta θ:            a network parameter (e.g. a weight w)
        //     eta η:              the learning rate
        //     gradient ∇J(x, y):  the gradient of the objective function,
        //                         i.e. the change for a specific theta θ

        // dbg!(&changes_to_w.W3);

        let x = [
            (&mut self.params.W1, changes_to_w.W1),
            (&mut self.params.W2, changes_to_w.W2),
            (&mut self.params.W3, changes_to_w.W3),
        ];

        for (weights, deltas) in x {
            for (row, row_deltas) in weights.iter_mut().zip(deltas) {
                for (w, delta) in row.iter_mut().zip(row_deltas) {
                    *w -= self.learn_rate * delta;
                }
            }
        }
    }

    fn compute_accuracy(&mut self, images: &[[f64; 784]], labels: &[[f64; 10]]) -> f64 {
        // This function does a forward pass of x, then checks if the indices
        // of the maximum value in the output equals the indices in the label
        // y. Then it sums over each prediction and calculates the accuracy.
        let mut correct_predictions = 0;
        // let mut predictions = Vec::with_capacity(images.len());

        for (image, label) in images.iter().zip(labels) {
            // let output = self.forward_pass(x);
            // let pred = np.argmax(output);
            // predictions.append(pred == np.argmax(y));
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
            // dbg!(pred, target);
            if pred == target {
                correct_predictions += 1;
            }
            // let result = if pred == target { 1.0 } else { 0.0 };
            // predictions.push(result);
        }

        println!("correct_predictions: {}", correct_predictions);

        correct_predictions as f64 / images.len() as f64

        // predictions.iter().sum::<f64>() / images.len() as f64
    }

    pub fn train(
        &mut self,
        train_images: &[[f64; 784]],
        train_labels: &[[f64; 10]],
        test_images: &[[f64; 784]],
        test_labels: &[[f64; 10]],
    ) {
        let start_time = Instant::now();
        for iteration in 0..self.epochs {
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
                self.update_network_parameters(changes_to_w);
                // dbg!(&self.params.W3);
                // println!("params changed: {}", old != self.params.W3);
            }

            let accuracy = self.compute_accuracy(test_images, test_labels);
            println!(
                "Epoch: {}, Time Spent: {}s, Accuracy: {}%",
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

fn outer_product(a: &Vec<f64>, b: &Vec<f64>) -> Matrix<f64> {
    // assert_eq!(a.len(), b.len());

    a.iter()
        .map(|a| b.iter().map(|b| a * b).collect())
        .collect()
}

fn transpose(matrix: &Matrix<f64>) -> Matrix<f64> {
    let old_row_size = matrix[0].len();
    let old_col_size = matrix.len();
    let new_row_size = old_col_size;
    let new_col_size = old_row_size;

    let mut transpose = Vec::with_capacity(new_col_size);

    for row in 0..new_col_size {
        transpose.push(Vec::with_capacity(new_row_size));
        for col in 0..new_row_size {
            transpose[row].push(matrix[col][row]);
        }
    }

    // Dimensions should have swapped.
    assert_eq!(transpose.len(), matrix[0].len());
    assert_eq!(transpose[0].len(), matrix.len());

    transpose
}

fn matrix_multiply(a: &Matrix<f64>, b: &Vec<f64>) -> Vec<f64> {
    assert_eq!(a[0].len(), b.len());

    a.iter()
        .map(|row| row.iter().zip(b).map(|(a, b)| a * b).sum())
        .collect()
}

fn sigmoid(x: &[f64], derivative: bool) -> Vec<f64> {
    if derivative {
        x.iter()
            .map(|&x| ((-x).exp()) / (((-x).exp() + 1.0).powf(2.0)))
            .collect()
    } else {
        x.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }
}

fn softmax(x: &[f64], derivative: bool) -> Vec<f64> {
    // Numerically stable with large exponentials

    let max = x.iter().copied().reduce(f64::max).unwrap();
    let exp_sum: f64 = x.iter().map(|&x| (x - max).exp()).sum();
    if derivative {
        x.iter()
            .map(|&x| (x - max).exp() / exp_sum * (1.0 - (x - max).exp() / exp_sum))
            .collect()
    } else {
        x.iter().map(|&x| (x - max).exp() / exp_sum).collect()
    }
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

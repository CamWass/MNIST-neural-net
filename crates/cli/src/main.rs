use std::fs;
use std::time::Instant;

use postcard::to_allocvec;

use data::*;
use network::{Accuracy, NeuralNet};

fn main() {
    let train_labels = get_train_labels();
    let test_labels = get_test_labels();
    let train_images = get_train_images();
    let test_images = get_test_images();

    println!("Loaded training data");

    let mut network = NeuralNet::new();

    const ITERATIONS: u8 = 30;

    let mut accuracy = Accuracy::default();

    let start_time = Instant::now();
    for iteration in 0..ITERATIONS {
        network.train(&train_images, &train_labels);
        accuracy = network.compute_accuracy(&test_images, &test_labels);
        println!(
            "Iteration: {}, Time elapsed: {}s, Accuracy: {}% ({}/{})",
            iteration,
            start_time.elapsed().as_secs(),
            accuracy.overall.percent_correct(),
            accuracy.overall.correct,
            accuracy.overall.total
        );
    }

    println!("Finished training");
    println!("Accuracy per element:");
    for (i, guesses) in accuracy.per_element.iter().enumerate() {
        println!(
            "  {}: {:.2}% ({}/{})",
            i,
            guesses.percent_correct(),
            guesses.correct,
            guesses.total
        );
    }

    let params = network.get_params();

    let params = to_allocvec(&params).unwrap();

    fs::write(
        format!(
            "pretrained/pretrained-params-{}",
            accuracy.overall.percent_correct().round() as usize
        ),
        params,
    )
    .unwrap();
}

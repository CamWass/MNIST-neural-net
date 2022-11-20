#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::fs;
use std::time::Instant;

use postcard::to_allocvec;

use network::*;

// const a: [u8; 100] = [0; 100];

// fn foo<const T: usize, const U: usize>() -> usize {
//     T * U
// }

fn parse_labels<const N: usize>(data: &[u8]) -> &[u8] {
    let magic_num_bits = [data[0], data[1], data[2], data[3]];
    let magic_num = u32::from_be_bytes(magic_num_bits);
    assert!(magic_num == 2049);

    let num_of_labels_bits = [data[4], data[5], data[6], data[7]];
    let num_of_labels = u32::from_be_bytes(num_of_labels_bits);
    assert!(num_of_labels as usize == N);

    &data[8..]
}

fn encode_labels(labels: &[u8]) -> Vec<[f32; 10]> {
    labels
        .iter()
        .map(|&l| {
            assert!(l <= 9);
            let mut a = [0.0; 10];
            a[l as usize] = 1.0;
            a
        })
        .collect()
}

fn as_chunks<T, const N: usize>(slice: &[T]) -> (&[[T; N]], &[T]) {
    assert_ne!(N, 0);
    let len = slice.len() / N;
    let (multiple_of_n, remainder) = slice.split_at(len * N);
    // SAFETY: We already panicked for zero, and ensured by construction
    // that the length of the subslice is a multiple of N.
    let array_slice = unsafe {
        // Inlined `as_chunks_unchecked` from nightly std

        // SAFETY: Caller must guarantee that `N` is nonzero and exactly divides the slice length
        let new_len = len;
        // SAFETY: We cast a slice of `new_len * N` elements into
        // a slice of `new_len` many `N` elements chunks.
        std::slice::from_raw_parts(multiple_of_n.as_ptr().cast(), new_len)
    };
    (array_slice, remainder)
}

// struct ImageData<'a>(&'a [u8]);

// impl<'a> IntoIterator for ImageData<'a> {
//     type IntoIter = ImageIterator<'a>;
//     type Item = Image<'a>;

//     fn into_iter(self) -> Self::IntoIter {
//         ImageIterator {
//             inner: self.0.array_chunks(),
//         }
//     }
// }

// struct ImageIterator<'a> {
//     inner: ArrayChunks<'a, u8, 784>,
// }

// impl<'a> Iterator for ImageIterator<'a> {
//     type Item = Image<'a>;
//     fn next(&mut self) -> Option<Self::Item> {
//         self.inner.next().map(|b| Image(b))
//     }
// }

fn parse_images<const N: usize, const S: usize>(data: &[u8]) -> &[[u8; S]] {
    let mut magic_num_bits = [0; 4];
    magic_num_bits.copy_from_slice(&data[..4]);
    let magic_num = u32::from_be_bytes(magic_num_bits);
    assert!(magic_num == 2051);

    let mut num_of_images_bits = [0; 4];
    num_of_images_bits.copy_from_slice(&data[4..8]);
    let num_of_images = u32::from_be_bytes(num_of_images_bits);
    assert!(num_of_images as usize == N);

    let mut num_of_rows_bits = [0; 4];
    num_of_rows_bits.copy_from_slice(&data[8..12]);
    let num_of_rows = u32::from_be_bytes(num_of_rows_bits);
    assert!(num_of_rows == 28);

    let mut num_of_columns_bits = [0; 4];
    num_of_columns_bits.copy_from_slice(&data[12..16]);
    let num_of_columns = u32::from_be_bytes(num_of_columns_bits);
    assert!(num_of_columns == 28);

    let image_data = as_chunks(&data[16..]);
    assert!(image_data.1.len() == 0);
    let image_data = image_data.0;
    assert_eq!(N, image_data.len());
    assert_eq!(S, image_data[0].len());

    image_data
}

fn encode_images(images: &[[u8; 784]]) -> Vec<[f32; 784]> {
    images
        .iter()
        .map(|image| {
            let mut a = [0.0; 784];
            for (i, &pixel) in image.iter().enumerate() {
                a[i] = (pixel as f32) / 255.0;
            }
            a
        })
        .collect()
}

fn main() {
    let train_label_data = fs::read("train-labels-idx1-ubyte").unwrap();
    let train_labels = parse_labels::<60_000>(&train_label_data);
    assert_eq!(&train_labels[..5], [5, 0, 4, 1, 9]);
    let train_labels = encode_labels(&train_labels);

    let test_label_data = fs::read("t10k-labels-idx1-ubyte").unwrap();
    let test_labels = parse_labels::<10_000>(&test_label_data);
    assert_eq!(&test_labels[..5], [7, 2, 1, 0, 4]);
    let test_labels = encode_labels(&test_labels);

    let train_image_data = fs::read("train-images-idx3-ubyte").unwrap();
    let train_images = parse_images::<60_000, 784>(&train_image_data);
    let train_images = encode_images(&train_images);

    let test_image_data = fs::read("t10k-images-idx3-ubyte").unwrap();
    let test_images = parse_images::<10_000, 784>(&test_image_data);
    let test_images = encode_images(&test_images);

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

trait Layer<const I: usize, const O: usize> {
    fn forwards(&mut self, input: &[u8; I]) -> &[u8; O];
    fn backwards(&mut self, input: &[u8; O]) -> &[u8; I];
}

struct InputLayer {}

struct HiddenLayer {}
struct OutputLayer {}

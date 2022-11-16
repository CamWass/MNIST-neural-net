#![feature(slice_as_chunks)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(dead_code)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use network::NeuralNet;

use std::fs;

// TODO: deduplicate with cli

fn parse_labels<const N: usize>(data: &[u8]) -> &[u8] {
    let magic_num_bits = [data[0], data[1], data[2], data[3]];
    let magic_num = u32::from_be_bytes(magic_num_bits);
    assert!(magic_num == 2049);

    let num_of_labels_bits = [data[4], data[5], data[6], data[7]];
    let num_of_labels = u32::from_be_bytes(num_of_labels_bits);
    assert!(num_of_labels as usize == N);

    &data[8..]
}

fn encode_labels(labels: &[u8]) -> Vec<[f64; 10]> {
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

    let image_data = &data[16..].as_chunks();
    assert!(image_data.1.len() == 0);
    let image_data = image_data.0;
    assert_eq!(N, image_data.len());
    assert_eq!(S, image_data[0].len());

    image_data
}

fn encode_images(images: &[[u8; 784]]) -> Vec<[f64; 784]> {
    images
        .iter()
        .map(|image| {
            let mut a = [0.0; 784];
            for (i, &pixel) in image.iter().enumerate() {
                a[i] = (pixel as f64) / 255.0;
            }
            a
        })
        .collect()
}

pub fn bench(c: &mut Criterion) {
    let train_label_data = include_bytes!("../../../train-labels-idx1-ubyte");
    let train_labels = &parse_labels::<60_000>(train_label_data)[..10_000];
    assert_eq!(&train_labels[..5], [5, 0, 4, 1, 9]);
    let train_labels = encode_labels(&train_labels);

    // let test_label_data = fs::read("t10k-labels-idx1-ubyte").unwrap();
    // let test_labels = parse_labels::<10_000>(&test_label_data);
    // assert_eq!(&test_labels[..5], [7, 2, 1, 0, 4]);
    // let test_labels = encode_labels(&test_labels);

    let train_image_data = include_bytes!("../../../train-images-idx3-ubyte");
    let train_images = &parse_images::<60_000, 784>(train_image_data)[..10_000];
    let train_images = encode_images(&train_images);

    // let test_image_data = fs::read("t10k-images-idx3-ubyte").unwrap();
    // let test_images = parse_images::<10_000, 784>(&test_image_data);
    // let test_images = encode_images(&test_images);

    let mut group = c.benchmark_group("train");
    group.sample_size(10);
    group.throughput(Throughput::Elements(train_images.len() as u64));

    group.bench_with_input("10k", &(&train_images, &train_labels), |b, input| {
        b.iter(|| {
            let mut network = NeuralNet::new_for_bench(black_box(1339));
            network.train(&train_images, &train_labels);
            black_box(network.get_params());
        });
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);

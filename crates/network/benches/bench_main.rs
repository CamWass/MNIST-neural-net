use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use data::*;
use network::NeuralNet;

pub fn bench(c: &mut Criterion) {
    let train_labels = get_train_labels();
    let train_images = get_train_images();

    let mut group = c.benchmark_group("train");
    group.sample_size(10);
    group.throughput(Throughput::Elements(10_000));

    let input = (&train_images[..10_000], &train_labels[..10_000]);
    group.bench_with_input("10k", &input, |b, (train_images, train_labels)| {
        b.iter(|| {
            let mut network = NeuralNet::new_for_bench(black_box(1339));
            network.train(train_images, train_labels);
            black_box(network.into_state());
        });
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);

static TRAIN_IMAGE_DATA: &[u8] = include_bytes!("../../../train-images-idx3-ubyte");
static TEST_IMAGE_DATA: &[u8] = include_bytes!("../../../t10k-images-idx3-ubyte");
static TRAIN_LABEL_DATA: &[u8] = include_bytes!("../../../train-labels-idx1-ubyte");
static TEST_LABEL_DATA: &[u8] = include_bytes!("../../../t10k-labels-idx1-ubyte");

pub fn get_train_images() -> Vec<[f32; 784]> {
    let train_images = parse_images::<60_000, 784>(&TRAIN_IMAGE_DATA);
    encode_images(&train_images)
}

pub fn get_test_images() -> Vec<[f32; 784]> {
    let test_images = parse_images::<10_000, 784>(&TEST_IMAGE_DATA);
    encode_images(&test_images)
}

pub fn get_train_labels() -> Vec<[f32; 10]> {
    let train_labels = parse_labels::<60_000>(&TRAIN_LABEL_DATA);
    debug_assert_eq!(&train_labels[..5], [5, 0, 4, 1, 9]);
    encode_labels(&train_labels)
}

pub fn get_test_labels() -> Vec<[f32; 10]> {
    let test_labels = parse_labels::<10_000>(&TEST_LABEL_DATA);
    debug_assert_eq!(&test_labels[..5], [7, 2, 1, 0, 4]);
    encode_labels(&test_labels)
}

fn parse_labels<const N: usize>(data: &[u8]) -> &[u8] {
    let magic_num_bits = [data[0], data[1], data[2], data[3]];
    let magic_num = u32::from_be_bytes(magic_num_bits);
    debug_assert!(magic_num == 2049);

    let num_of_labels_bits = [data[4], data[5], data[6], data[7]];
    let num_of_labels = u32::from_be_bytes(num_of_labels_bits);
    debug_assert!(num_of_labels as usize == N);

    &data[8..]
}

fn encode_labels(labels: &[u8]) -> Vec<[f32; 10]> {
    labels
        .iter()
        .map(|&l| {
            debug_assert!(l <= 9);
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
    debug_assert!(magic_num == 2051);

    let mut num_of_images_bits = [0; 4];
    num_of_images_bits.copy_from_slice(&data[4..8]);
    let num_of_images = u32::from_be_bytes(num_of_images_bits);
    debug_assert!(num_of_images as usize == N);

    let mut num_of_rows_bits = [0; 4];
    num_of_rows_bits.copy_from_slice(&data[8..12]);
    let num_of_rows = u32::from_be_bytes(num_of_rows_bits);
    debug_assert!(num_of_rows == 28);

    let mut num_of_columns_bits = [0; 4];
    num_of_columns_bits.copy_from_slice(&data[12..16]);
    let num_of_columns = u32::from_be_bytes(num_of_columns_bits);
    debug_assert!(num_of_columns == 28);

    let image_data = as_chunks(&data[16..]);
    debug_assert!(image_data.1.len() == 0);
    let image_data = image_data.0;
    debug_assert_eq!(N, image_data.len());
    debug_assert_eq!(S, image_data[0].len());

    image_data
}

pub fn encode_image(image: &[u8; 784], output: &mut [f32; 784]) {
    for (i, &pixel) in image.iter().enumerate() {
        output[i] = (pixel as f32) / 255.0;
    }
}

fn encode_images(images: &[[u8; 784]]) -> Vec<[f32; 784]> {
    images
        .iter()
        .map(|image| {
            let mut a = [0.0; 784];
            encode_image(image, &mut a);
            a
        })
        .collect()
}

pub fn as_chunks<T, const N: usize>(slice: &[T]) -> (&[[T; N]], &[T]) {
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

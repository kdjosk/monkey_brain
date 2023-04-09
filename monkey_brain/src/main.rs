use monkey_brain::{Net, SGD, load_data, one_hot};
use ndarray::Array2;

fn main() {
    let train_set = load_data("train").unwrap();
    let test_set = load_data("t10k").unwrap();

    let train_set: Vec<_> = train_set
        .into_iter()
        .map(|img| (img.image, one_hot(img.label as usize)))
        .collect();

    let test_set: Vec<_> = test_set
        .into_iter()
        .map(|img| (img.image, img.label as usize))
        .collect();

    let mut net = Net::new(vec![784, 30, 10]);
    net.forward_pass(&train_set[0].0);

    let sgd = SGD::new(
        train_set,
        test_set,
        10,
        30,
        3.0,
    );

    sgd.train(&mut net);

}


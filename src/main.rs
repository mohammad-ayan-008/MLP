use core::f64;
use std::result;
mod mlp;
use nalgebra::{DVector, Matrix1x2, Matrix4x2};
// 1 row 2 coln
pub struct Perceptron {
    weight: Matrix1x2<f64>,
    bias: f64,
    learning_rate: f64,
}
impl Perceptron {
    fn new() -> Self {
        Self {
            weight: Matrix1x2::from_element(1.0),
            bias: 0.0,
            learning_rate: 0.01,
        }
    }
    fn predict(&self, x: &Matrix1x2<f64>) -> f64 {
        let result = self.weight * x.transpose();
        self.activation(result[(0, 0)] + self.bias)
    }
    fn activation(&self, val: f64) -> f64 {
        if val >= 0.0 { 1.0 } else { 0.0 }
    }

    fn train(&mut self, input: &Matrix4x2<f64>, lable: &DVector<f64>, repetetion: i32) {
        for _ in 0..repetetion {
            for i in 0..input.nrows() {
                let row = input.row(i);
                let z = (self.weight * row.transpose())[(0, 0)] + self.bias;
                if lable[i] * z < 1.0 {
                    self.weight[(0, 0)] += lable[i] * row[0] * self.learning_rate;
                    self.weight[(0, 1)] += lable[i] * row[1] * self.learning_rate;
                    self.bias += self.learning_rate * lable[i];
                }
            }
        }
        println!("weights = {}  bias={} \n", self.weight, self.bias);
    }
}

fn main() {
    let mut perceptron = Perceptron::new();
    let training_data = Matrix4x2::from_row_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    mlp::main();
    // println!("{:?}", perceptron.weight);
    // let answer = DVector::from_vec(vec![-1.0, -1.0, -1.0, 1.0]); // Hing Loss is designed for -1 and 1 only
    // perceptron.train(&training_data, &answer, 1000);
    //
    // println!("Trained for given data set \n");
    // println!(
    //     "0 AND 1 is {}",
    //     perceptron.predict(&Matrix1x2::from_row_slice(&[0.0, 1.0]))
    // );
    // println!(
    //     "0 AND 0 is {}",
    //     perceptron.predict(&Matrix1x2::from_vec(vec![0.0, 0.0]))
    // );
    // println!(
    //     "1 AND 0 is {}",
    //     perceptron.predict(&Matrix1x2::from_vec(vec![1.0, 0.0]))
    // );
    // println!(
    //     "1 AND 1 is {}",
    //     perceptron.predict(&Matrix1x2::from_vec(vec![1.0, 1.0]))
    // );
}

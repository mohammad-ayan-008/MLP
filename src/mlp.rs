use core::f64;
use std::vec;

use nalgebra::{DVector, Matrix1x2, Matrix2, Matrix2x1, Matrix4x2};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[derive(Debug, PartialEq, Clone)]
struct Perceptron {
    weight: Matrix2x1<f64>,
    bias: f64,
}
struct Perceptron_hiden {
    weight: Matrix2<f64>,
    bias: Matrix2x1<f64>,
}

impl Perceptron {
    fn new_random(rng: &mut StdRng) -> Self {
        let w1 = rng.random_range(-1.0..1.0);
        let w2 = rng.random_range(-1.0..1.0);
        let b = rng.random_range(-1.0..1.0);
        Perceptron {
            weight: Matrix2x1::new(w1, w2),
            bias: b,
        }
    }
}

struct Layer {
    hidden_l: Perceptron_hiden,
    output_l: Perceptron,
}

impl Layer {
    fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(42); // fixed seed for reproducibility
        let out = Perceptron::new_random(&mut rng);
        let hidden_layer = Perceptron_hiden {
            weight: Matrix2::from_row_slice(&[
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ]),
            bias: Matrix2x1::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)),
        };

        Layer {
            hidden_l: hidden_layer,
            output_l: out,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn loss(&self, ans: f64, pred: f64) -> f64 {
        let eps = 1e-10;
        let p = pred.clamp(eps, 1.0 - eps);
        -(ans * p.ln() + (1.0 - ans) * (1.0 - p).ln())
    }

    pub fn forward_train(
        &mut self,
        input: &Matrix4x2<i32>,
        answers: &DVector<i32>,
        learning_rate: f64,
    ) {
        for _epoch in 0..20000 {
            for i in 0..input.nrows() {
                // Forward pass
                let x: Matrix2x1<f64> = nalgebra::convert(input.row(i).transpose().into_owned());

                let output_hidden = self.hidden_l.weight.transpose() * x + self.hidden_l.bias;

                let out_h_sig = output_hidden.map(Self::sigmoid);

                let y_pred = Self::sigmoid(
                    (self.output_l.weight.transpose() * out_h_sig)[(0, 0)] + self.output_l.bias,
                );
                let y_true = answers[i] as f64;
                let _loss = self.loss(y_true, y_pred);

                let out_error = y_pred - y_true;

                self.output_l.weight -= learning_rate * out_error * out_h_sig;
                self.output_l.bias -= learning_rate * out_error;

                let mut dever_out = Matrix1x2::new(0.0, 0.0);
                for i in out_h_sig.iter().enumerate() {
                    dever_out[i.0] = out_h_sig[i.0] * (1.0 - out_h_sig[i.0]);
                }

                self.hidden_l.weight -=
                    learning_rate * out_error * self.output_l.weight * dever_out * y_true;
            }
        }
    }

    pub fn predict(&self, x: Matrix2x1<f64>) -> f64 {
        let output_hidden = self.hidden_l.weight.transpose() * x + self.hidden_l.bias;
        let out = output_hidden.map(Self::sigmoid);
        Self::sigmoid((self.output_l.weight.transpose() * out)[(0, 0)] + self.output_l.bias)
    }
}
pub fn main() {
    let inp = Matrix4x2::from_row_slice(&[0, 0, 0, 1, 1, 0, 1, 1]);
    let ans = DVector::from_vec(vec![0, 1, 1, 0]);

    let mut layer = Layer::new();
    layer.forward_train(&inp, &ans, 0.01);

    for &(a, b) in &[(0, 0), (0, 1), (1, 0), (1, 1)] {
        let prob = layer.predict(Matrix2x1::from_row_slice(&[a as f64, b as f64]));
        let class = if prob > 0.5 { 1 } else { 0 };
        println!("Input ({}, {}) => prob: {:.4},  {}", a, b, prob, class);
    }
}

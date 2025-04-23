use std::vec;

use rand::{thread_rng, Rng};
use nalgebra::{DMatrix, DVector, Matrix1x2, Matrix4x2, Vector2};

#[derive(Debug, PartialEq, Clone)]
struct Perceptron {
    weight: Matrix1x2<f64>,
    bias: f64,
}

impl Perceptron {
    /// Create a neuron with random weights and bias to break symmetry
    fn new_random() -> Self {
        let mut rng = thread_rng();
        let w1 = rng.gen_range(-1.0..1.0);
        let w2 = rng.gen_range(-1.0..1.0);
        let b = rng.gen_range(-1.0..1.0);
        Perceptron { weight: Matrix1x2::new(w1, w2), bias: b }
    }
}

struct Layer {
    hidden_l: [Perceptron; 2],
    output_l: Perceptron,
}

impl Layer {

    fn new() -> Self {
        Self {
            hidden_l: [Perceptron::new_random(), Perceptron::new_random()],
            output_l: Perceptron::new_random(),
        }
    }


    fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }


    fn loss(&self, ans: f64, pred: f64) -> f64 {
        let eps = 1e-10;
        let p = pred.clamp(eps, 1.0 - eps);
        -(ans * p.ln() + (1.0 - ans) * (1.0 - p).ln())
    }


    pub fn forward_train(&mut self,
                         input: &Matrix4x2<i32>,
                         answers: &DVector<i32>,
                         learning_rate: f64) {
        for epoch in 0..1000 {
            for i in 0..input.nrows() {
                // Forward pass
                let row = input.row(i);
                let x = Vector2::new(row[0] as f64, row[1] as f64);

                let mut hidden_out = Vector2::zeros();
                for (j, neuron) in self.hidden_l.iter().enumerate() {
                    let z = neuron.weight.dot(&x.transpose()) + neuron.bias;
                    hidden_out[j] = Self::sigmoid(z);
                }
                let z_o = self.output_l.weight.dot(&hidden_out.transpose()) + self.output_l.bias;
                let y_pred = Self::sigmoid(z_o);
                let y_true = answers[i] as f64;

                if epoch % 500 == 0 && i == 0 {
                    println!("Epoch {}: loss = {:.6}", epoch, self.loss(y_true, y_pred));
                }


                let output_error = y_pred - y_true;

                let old_w_o = self.output_l.weight;


                let mut hidden_error = Vector2::zeros();
                for j in 0..self.hidden_l.len() {
                    let dh = hidden_out[j] * (1.0 - hidden_out[j]);
                    hidden_error[j] = old_w_o[j] * output_error * dh;
                }


                self.output_l.weight -= learning_rate * output_error * hidden_out.transpose();
                self.output_l.bias -= learning_rate * output_error;


                for j in 0..self.hidden_l.len() {
                    self.hidden_l[j].weight -= learning_rate * hidden_error[j] * x.transpose();
                    self.hidden_l[j].bias -= learning_rate * hidden_error[j];
                }
            }
        }
    }

 
    pub fn predict(&self, x_i: Vector2<i32>) -> f64 {
        let x = x_i.map(|f| f as f64);
        let mut hidden_out = Vector2::zeros();
        for (j, neuron) in self.hidden_l.iter().enumerate() {
            let z = neuron.weight.dot(&x.transpose()) + neuron.bias;
            hidden_out[j] = Self::sigmoid(z);
        }
        let z_o = self.output_l.weight.dot(&hidden_out.transpose()) + self.output_l.bias;
        Self::sigmoid(z_o)
    }
}

pub fn main() {
    let inp = Matrix4x2::from_row_slice(&[0, 0, 0, 1, 1, 0, 1, 1]);
    let ans = DVector::from_vec(vec![0, 1, 1, 0]);

    let mut layer = Layer::new();
    layer.forward_train(&inp, &ans, 0.1);

    for &(a, b) in &[(0, 0), (0, 1), (1, 0), (1, 1)] {
        let prob = layer.predict(Vector2::new(a, b));
        let class = if prob > 0.5 { 1 } else { 0 };
        println!("Input ({}, {}) => prob: {:.4},  {}", a, b, prob, class);
    }
}


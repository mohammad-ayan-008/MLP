use core::f64;

use nalgebra::{ DVector, Dyn, Matrix1, Matrix1x2, Matrix1x4, Matrix2x1, Matrix4x2, Storage};
              // 1 row 2 coln
struct Perceptron{
    weight:Matrix1x2<f64>,
    bias:f64,
    learning_rate:f64
}
impl Perceptron {
    fn new(input_size:i32)-> Self{
        Self{
            weight: Matrix1x2::from_element(0.0),
            bias:0.0,
            learning_rate: 0.01
        }
    }
    fn predict(&self,x:&Matrix1x2<f64>)->f64{
        let result = self.weight * x.transpose();
        self.activation(result[(0,0)]+self.bias)
    }
    fn activation(&self, val:f64)->f64{
        if val >= 0.0 {
             1.0
        }else {
            0.0
        }
    }

    fn train(&mut self,input:&Matrix4x2<f64>,lable:&Matrix1x4<f64>,repetetion:i32){
        for _ in 0..repetetion{
            for i in 0..input.nrows(){
                let input_row= input.row(i);
                let y_predict = self.predict(&input_row.into());
                self.weight += self.learning_rate * (lable[i]- y_predict) * input_row;
                self.bias +=  self.learning_rate * (lable[i] - y_predict)

            } 
        }
         println!("weights = {}  bias={} \n",self.weight,self.bias);
    }
}

fn main() {
    let mut perceptron = Perceptron::new(2); 
    let training_data = Matrix4x2::from_row_slice(&[
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ] );

    println!("{:?}",perceptron.weight);
    let answer = Matrix1x4::from_vec(vec![0.0 , 1.0 , 1.0, 1.0]);
     perceptron.train(&training_data, &answer,1000);
   

    println!("Trained for given data set \n");
    println!("0 OR 1 is {}",perceptron.predict(&Matrix1x2::from_row_slice(&[0.0,1.0])));
    println!("0 OR 0 is {}",perceptron.predict(&Matrix1x2::from_vec(vec![0.0,0.0])));
    println!("1 OR 0 is {}",perceptron.predict(&Matrix1x2::from_vec(vec![1.0,0.0])));
    println!("1 OR 1 is {}",perceptron.predict(&Matrix1x2::from_vec(vec![1.0,1.0])));
}

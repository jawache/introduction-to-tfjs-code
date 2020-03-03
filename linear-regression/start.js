/** GLOBAL SETTINGS **/
let LOSS = 0;
let CURRENT_EPOCH = 0;
const MAX_EPOCHS = 300;

// This will store mouse x,y points that have been scaled from 0->1
let Xs = [];
let Ys = [];

// The equation of a line
let A = -0.4;
let C = 1;
const getY = x => A * x + C;


// Create tensor variables to store the weights of `A` and `C` 
// TODO: Replace A and C with Math.random() when running properly
const a = tf.variable(tf.scalar(A));
const c = tf.variable(tf.scalar(C));

// Setup the optimiser
const learningRate = 0.5;

// Crete an optimiser, this will be used to change the weights (m and c) to minimise the loss function
const optimizer = tf.train.sgd(learningRate);

function predict(x) {
  // y = m * x + b
  // TODO
}

function loss(predictedYs, actualYs) {
  // Mean Squared Error
  // TODO
}

async function train() {
  if (Xs.length) {
    for (CURRENT_EPOCH = 0; CURRENT_EPOCH < MAX_EPOCHS; CURRENT_EPOCH++) {
      // TODO
    }
  }
}
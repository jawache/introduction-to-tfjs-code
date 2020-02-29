/**
 * The equation for a polynomial (curved line) is this
 *
 * Y = A*X^2 + B*X + C
 *
 * Very similar to the linear regression example previously except now there are 3 variables to optimise, A, B and C instead of just A + C
 *
 *
 * See if you can change this file so it fits a polynomial curve instead.
 */

/** GLOBAL SETTINGS **/
let LOSS = 0;
let CURRENT_EPOCH = 0;
const MAX_EPOCHS = 300;

// This will store mouse x,y points that have been scaled from 0->1
let Xs = [];
let Ys = []

// The equation of a curve
let A = -0.4;
// TODO: Maybe we need to store another coefficient here?
let C = 0.5;
const getY = x => A * x + C; // TODO: Hmm is this the equation of a curve?

// Create tensor variables to store the weights of A, B and C
const a = tf.variable(tf.scalar(Math.random()));
//TODO: Maybe another tensorflow variable here?
const c = tf.variable(tf.scalar(Math.random()));

// Setup the optimiser
const learningRate = 0.5;

// Crete an optimiser, this will be used to change the weights (m and c) to minimise the loss function
const optimizer = tf.train.sgd(learningRate);

// Is passed in an array of X values and returns an array of predicted Y values based on the current values of a, b and c weights
function predict(x) {
  // TODO: This might need changing to be a polynomial equation, HINT - x.square() squares x
  return a.mul(x).add(c);
}

// When passed in the array of predictedYs calculates the mean square loss compared to the actualYs
function loss(predictedYs, actualYs) {
  // Mean Squared Error
  let x = predictedYs
    .sub(actualYs)
    .square()
    .mean();
  LOSS = x.dataSync()[0];
  return x;
}

// Pass in the actualXs and the actualYs (from the mouse clicks)
// use the actualXs to calculate the prdictedYs
// pass predictedYs and actualYs to the optimiser and try to minimise that value
async function train(numIterations = 1) {
  if (Xs.length) {
    for (CURRENT_EPOCH = 0; CURRENT_EPOCH < numIterations; CURRENT_EPOCH++) {
      tf.tidy(() => {
        const actualXs = tf.tensor(Xs, [Xs.length, 1]);
        const actualYs = tf.tensor(Ys, [Ys.length, 1]);

        optimizer.minimize(() => {
          let predictedYs = predict(actualXs);
          return loss(predictedYs, actualYs);
        });

        A = a.dataSync()[0];
        // TODO: Maybe we need to extract the value from another tf.variable here?
        C = c.dataSync()[0];
        // console.log(A, B, C);
      });
      await tf.nextFrame();
    }
  }
}
/** GLOBAL SETTINGS **/
var LOSS = 0;

// This will store mouse x,y points
let Xs = [];
let Ys = [];

// The equation of a line
let A = -0.3;
let C = 0.5;
const getY = x => A * x + C;


// Manual Loss Function
function manual_loss() {
  let mean = 0;
  const size = Xs.length;
  for (let i = 0; i < size; i++) {
    const x = Xs[i];
    const y = Ys[i];
    const predictedY = getY(x);
    mean = mean + Math.pow(y - predictedY, 2);
  }
  LOSS = mean / size;
}

// TensorFlow Loss Function
function tf_loss() {
  const actualXs = tf.tensor(Xs, [Xs.length, 1]);
  const actualYs = tf.tensor(Ys, [Ys.length, 1]);

  const a = tf.scalar(A);
  const c = tf.scalar(C);
  predictedYs = a.mul(actualXs).add(c);

  let x = predictedYs
    .sub(actualYs)
    .square()
    .mean();

  LOSS = x.dataSync()[0];
}

async function train() {
  manual_loss()
}
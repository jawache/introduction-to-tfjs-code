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
  // TODO
}

// TensorFlow Loss Function
function tf_loss() {
  // TODO
}

async function train() {
  manual_loss()
}
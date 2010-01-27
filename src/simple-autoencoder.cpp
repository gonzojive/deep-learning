#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Array>
#include <vector>
#include <math.h>

USING_PART_OF_NAMESPACE_EIGEN

using namespace std;


#define WINDOW_PIXEL_SIZE 8 
#define DEFAULT_ALPHA 0.1
#define WEIGHT_DECAY 0.002
#define RUNNING_AVERAGE_DECAY 0.999
#define TARGET_ACTIVATION -.996
#define DEFAULT_BETA 5.0

double beta = DEFAULT_BETA;
double alpha = DEFAULT_ALPHA;
char * oFilename;
int numIterations=40000;
void activationFunction(const MatrixXd& x, MatrixXd & result);
void generateTrainingExample(vector<MatrixXd> & images, MatrixXd & output);
void hamProd(const MatrixXd & a, const MatrixXd & b, MatrixXd & output);

void printMatMeta(const MatrixXd & a) {
  cout << a.rows() << "x" << a.cols();
}
void printMat(const MatrixXd & a, ostream & out) {
  out << "";
  int rows = a.rows();
  int cols = a.cols();
  for (int row=0; row < rows; row++) {
    for (int col=0; col < cols; col++) {
      out << a(row,col);
      if (col != cols - 1)
	out << " ";
    }
    if (row == rows-1)
      out << "" << endl;
    else
      out << "" << endl;
  }
}

// Returns a random positive integer in [0,lim-1]
int randInt(int lim);
double randDouble(double lim);

class NeuralNetwork {
protected:
  // _W[l](i, j) = W^(l)_ij from notes.  that is, ij The parameter
  // associated with the connection between unit j in layer l, and
  // unit i in layer l + 1.  In other words, row ``unit'' in W[layer]
  // corresponds to how much the inputs from layer ``layer - 1''
  // should be weighted going into unit # ``unit'' in layer ``layer''
  // Thus, row x * activations in x-1 = scalar result.
  vector<MatrixXd> _W;
  /// _activations[i] is a vector such that
  /// _activations[layerNum](unitNum) is a scalar value corresponding
  /// to the activation value for a particular "neuron"
  vector<MatrixXd> _activations; 
  vector<MatrixXd> _activationFnInputs;
  vector<MatrixXd> _activationAverages; 

  /// _biases is a vector such that _biases[layerNum][unitNum]
  /// corresponds to the bias term for a particular neuron in the
  /// network
  vector<MatrixXd> _biases; 

public:
  /** Constructs a nueral network with a given number of layers and a
      given number of neurons for each particular layer */
  NeuralNetwork(int numLayers, int * unitCounts) {
    // initialize the activations and biases
    for (int layer=0; layer < numLayers; layer++) {
      //MatrixXd layerActivations = MatrixXd(unitCounts[layer], 1);
      //layerActivations.setZero();
      _activations.push_back(MatrixXd(unitCounts[layer], 1));//layerActivations);
      _activationFnInputs.push_back(MatrixXd(unitCounts[layer], 1));
      _activationAverages.push_back(MatrixXd(unitCounts[layer], 1).setZero());

    }
    // initialize the weights/biases
    for (int i=1; i < numLayers; i++) {
      // biases -> 0
      MatrixXd layerBias = MatrixXd(unitCounts[i], 1);
      layerBias.setZero();
      _biases.push_back(layerBias);

      // weights -> 1 / (sqrt(fanout)) where fanout = incoming degree of node
      int rows = unitCounts[i];
      int cols = unitCounts[i - 1];
      
      MatrixXd layerWeightMatrix = MatrixXd(rows, cols);

      for (int row=0; row < rows; row++) {
	double fanout = static_cast<double>(unitCounts[i - 1]);
	for (int col=0; col < cols; col++) {
	  layerWeightMatrix(row, col) = randDouble(1.0 / (sqrt(fanout) + .0000000001));
	}
      }
      
      _W.push_back(layerWeightMatrix);
    }
  }
  
  /// same as connectionWeight(l,i,j)
  double W(int l, int i, int j) {
    return connectionWeight(l,i,j);
  }

  /// The parameter associated with the connection between unit j in
  /// layer l, and unit i in layer l + 1. (i.e. connectionWeight)
  double connectionWeight(int l, int i, int j) {
    return _W[l](i,j);
  }

  /// Returns the activation vector of coefficients for 
  MatrixXd & layerActivationVector(int layer) {
    return _activations[layer];
  }

  
  void setW(int l, int i, int j, double value) {
    setConnectionWeight(l,i,j,value);
  }

  void setConnectionWeight(int layer, int layeriPlus1Index, int layerjIndex, double value) {
    _W[layer](layeriPlus1Index,layerjIndex) = value;
  }

  // Number of ``neurons'' in layer # layer
  int layerSize(int layer) {
    return _activations[layer].size();
  }

  // number of layers in the neural net
  int layerCount() {
    return _activations.size();
  }

  /// Given a set of inputs, fwdpropagates them into the neural
  /// network and stores the output in outNN.  input is an
  /// INPUT_DIMESION x 1 matrix of values.
  void forwardPropagateActivations(const MatrixXd & input ) {
    // set the first layer of activation values to the input values directly
    _activations[0] = input;
   
    for (int layer=1; layer < _activations.size(); layer++) {
      _activations[layer] = _W[layer - 1] * _activations[layer - 1];
      _activations[layer] = _activations[layer] + _biases[layer - 1];
      //MatrixXd & mlayer = _activations[layer];
      _activationFnInputs[layer] = _activations[layer];

      activationFunction(_activationFnInputs[layer], _activations[layer]);
    }
    
  }

  void sparseLearningPass(bool debugOutput ) {
    // update the activation values and averages for all internal layers (not input and output)
    for (int layer=1; layer < layerCount() - 1; layer++) {
      // update the activation values for this layer
      _activationFnInputs[layer] = _W[layer - 1] * _activations[layer - 1] + _biases[layer - 1];
      activationFunction(_activationFnInputs[layer], _activations[layer]);
      // update the average activation for this layer
      _activationAverages[layer] = _activations[layer] * (1.0 - RUNNING_AVERAGE_DECAY)
	+ _activationAverages[layer] * RUNNING_AVERAGE_DECAY;

      // update the biases
      double averageBiasAdjustment = 0;
      MatrixXd & biasVector = _biases[layer - 1];
      for (int unit=0; unit < biasVector.size(); unit++) {
	double biasAdjustment = (-1.0 * alpha * beta) * ( _activationAverages[layer](unit) - TARGET_ACTIVATION);
	averageBiasAdjustment += biasAdjustment;
	biasVector(unit, 0) = biasVector(unit, 0) + biasAdjustment;
      }
      if (debugOutput) {
	averageBiasAdjustment = averageBiasAdjustment / static_cast<double>(biasVector.size());
	cout << "Average bias adjustment: " << averageBiasAdjustment << endl;
      }
    }
  }

  
  void backprop(const MatrixXd & trainingOutput) {
    vector<MatrixXd> gammas;
    for (int i=0; i < layerCount(); i++)
      gammas.push_back(MatrixXd()); // note: gammas[0] and gammas[1] is invalid but we keep them around for syntactic convenience

    // 1. For the output layer, set gamma = -(trainingOutput - activation[layer]) 
    MatrixXd & lastLayerGamma = gammas[layerCount() - 1];
    MatrixXd activationPrime;
    activationFunctionPrime(layerCount() - 1, activationPrime);

    hamProd( (trainingOutput - _activations[layerCount() - 1]) * -1.0,
	     activationPrime, 
	     lastLayerGamma);
    // 2. For each other layer, set gamma = ((_W[layer]Transpose * gamma[layer+1]) .* f'(activationFnInputs[layer])
    for (int layer=layerCount() - 2; layer > 0; layer--) {
      MatrixXd & layerGamma = gammas[layer];
      MatrixXd & fwdLayerGamma = gammas[layer+1];
      activationFunctionPrime(layer, activationPrime);
      hamProd( _W[layer].transpose() * fwdLayerGamma, activationPrime, layerGamma);
      
    }

    // 3. Update parameters according to the gammas
    for (int layer=1; layer < layerCount(); layer++) {
      MatrixXd A = gammas[layer] * _activations[layer - 1].transpose();
      MatrixXd B = (_W[layer - 1] * WEIGHT_DECAY);
      MatrixXd weightChange = (A + B) * (alpha);
      //(( gammas[layer] * _activations[layer - 1].transpose())
      // + (_W[layer - 1] * WEIGHT_DECAY)) * ALPHA;
      _W[layer-1] = _W[layer-1] - weightChange;

      MatrixXd & E = _biases[layer-1];
      MatrixXd F = gammas[layer] * alpha;
      _biases[layer-1] = (E - F);
    }

  }

  void activationFunctionPrime(int layer, MatrixXd & output) {
    int lSize = layerSize(layer);
    output.resize(lSize, 1);
    for (int i = 0; i < lSize; i++) {
      double activation = _activations[layer](i, 0);
      output(i, 0) = 1.0 - activation*activation;
    }
  }

  double computeAverageActivation(int layer) {
    double sum = 0.0;
    for (int i=0; i < _activations[layer].size(); i++)
      sum += _activationAverages[layer](i,0);//_activations[layer](i,0);
    return sum / static_cast<double>(_activations[layer].size());
  }
   
  double computeAverageActivationInput(int layer) {
    double sum = 0.0;
    for (int i=0; i < _activations[layer].size(); i++)
      sum += _activationFnInputs[layer](i,0);//_activations[layer](i,0);
    return sum / static_cast<double>(_activations[layer].size());
  }
   
  double computeCost(const MatrixXd & input, const MatrixXd & correctOutput) {
    forwardPropagateActivations(input);
    MatrixXd & nnOutput = _activations[layerCount() - 1];
    double halfSquaredError = (nnOutput - correctOutput).squaredNorm() * .5;
    double decay =0;
    for (int layer=0; layer < layerCount() - 1; layer++) {
      decay += _W[layer].squaredNorm();
    }
    decay *= WEIGHT_DECAY * .5;
    double cost = halfSquaredError - decay;
    cout << "error - decay = " << halfSquaredError << " - " << decay << " = " << cost << " (avg. activation: " << computeAverageActivation(1) << ", z: " << computeAverageActivationInput(1) << ")" << endl;
    //cout << "activations: ";
    //printMat(_activations[1].transpose(), cout);

    return cost;
    
  }

  void printOutputForVisualizer() {
    if (!oFilename) {
      printMat(_W[0], cout);
    }
    else {
      ofstream out;
      out.open (oFilename);
      printMat(_W[0], out);
      out.close();
    }

    
  }

      
  
  
};

int main(int argc, char* argv[])
{
  cout << "Reading in the input data" << endl;
  /**************  Read in images *****************/
  int imgSize = 512;
  int numImages = 10;
  string filename = "olsh.dat";

  vector<MatrixXd> images;
     
  ifstream indata; 
  double num; // variable for input value
  
  indata.open(filename.c_str());
  if(!indata) { 
    cerr << "Error: file could not be opened" << endl;
    return 1;
   }
  
  for(int i = 0; i < numImages; ++i) {
    MatrixXd m(imgSize,imgSize);
    for(int r = 0; r < imgSize; ++r) { 
      for(int c = 0; c < imgSize; ++c) {
	if(indata.eof()) {
	  cerr << "Error: ran out of input values on (" << r << "," << c << ")" << endl;
	  return 1;
	}
	
	indata >> num;
	m(r,c) = num;
      }
    }
    images.push_back(m);
  }
  indata.close();
  
  cout << "Input data loaded" << endl;


  /*************** YOUR CODE HERE ************/
  /* parse the command line options */
  oFilename = NULL;
  if (argc > 1) {
    sscanf(argv[1], "%lf", &alpha);
  }
  if (argc > 2) {
    sscanf(argv[2], "%d", &numIterations);
  }
  if (argc > 3) {
    sscanf(argv[3], "%lf", &beta);
  }
  if (argc > 4) {
    oFilename = argv[4];
    cout << "Outputting to file " << oFilename << endl;
  }
  else {
    oFilename = (char *)malloc(256);
    sprintf(oFilename, "bases/alpha%fnumIterations%d.dat", alpha, numIterations);
  }
  /* For each training example: 

     (i) Run a forward pass on our network on input x, to compute all
     units? activations;

     (ii) Perform one step of stochas- tic gradient descent using
     backpropagation;
     
     (iii) Perform the updates given in Equations (8-9).
  */
  int unitCounts[] = {WINDOW_PIXEL_SIZE * WINDOW_PIXEL_SIZE, 
		      30, 
		      WINDOW_PIXEL_SIZE * WINDOW_PIXEL_SIZE};
  NeuralNetwork nn = NeuralNetwork(3, unitCounts);
  
  MatrixXd trainingImage;
  cout << "Performing " << numIterations << " iterations with alpha = " << alpha << endl;
  for (int iteration=0; iteration < numIterations; iteration++) {
    generateTrainingExample(images, trainingImage);

    if (iteration % 1000 == 0 || iteration < 11) {
      cout << iteration << " " << flush;
      cout << endl << "   ";
      nn.computeCost(trainingImage, trainingImage);
      
    }
    // (i) run a feedforward pass on our network
    nn.forwardPropagateActivations(trainingImage);
    
    // (ii) perform one step of stochastic gradient descent
    nn.backprop(trainingImage);

    // (iii) perform the updates given in Equations (8-9).
    nn.sparseLearningPass(iteration % 1000 == 0 || iteration < 11);

    if (iteration % 1000 == 0 || iteration < 11) {
      cout << "   ";
      nn.computeCost(trainingImage, trainingImage);
    }

  }
  cout  << "." << endl;

  cout << "Finished training the neural network." << endl;

  nn.printOutputForVisualizer();
  

  
  
  
  return 0;
}

void randomSubmatrix(MatrixXd & matrix, int rows, int cols, MatrixXd & output) {
  output.resize(rows, cols);
  // chose random start row and col
  int startRow = randInt(matrix.rows() - rows);
  int startCol = randInt(matrix.cols() - cols);

  for (int row=startRow; row < startRow + rows; row++) {
    for (int col=startCol; col < startCol + cols; col++) {
      output(row - startRow, col - startCol) = matrix(row, col);
    }
  }
}

/** Given a matrix, puts all of its values into a single column of the output vector
 **/
void vectorizeMatrix(MatrixXd & matrix, MatrixXd & vector) {
  vector.resize(matrix.rows() * matrix.cols(), 1);
  int pixelNumber=0;
  for (int row=0; row < matrix.rows(); row++) {
    for (int col=0; col < matrix.cols(); col++) {
      vector(pixelNumber, 0) = matrix(row, col);
      ++pixelNumber;
    }
  }
}

void generateTrainingExample(vector<MatrixXd> & images, MatrixXd & output) {
  // choose random image
  MatrixXd & img = images[randInt(images.size())];
  // take random 8x8 window of the matrix
  MatrixXd window = MatrixXd(WINDOW_PIXEL_SIZE, WINDOW_PIXEL_SIZE);
  randomSubmatrix(img, WINDOW_PIXEL_SIZE, WINDOW_PIXEL_SIZE, window);
  // convert that into a vector
  vectorizeMatrix(window, output);
}

void activationFunction(const MatrixXd& x, MatrixXd& result) {
  result.resize(x.rows(),x.cols());

  for(int r = 0; r < x.rows(); r++)
    for(int c = 0; c < x.cols(); c++)
      result(r,c) = tanh(x(r,c));
}

// element-wise product of a and b
void hamProd(const MatrixXd & a, const MatrixXd & b, MatrixXd & output) {
  assert(a.rows() == b.rows());
  assert(a.cols() == b.cols());
  output.resize(a.rows(), a.cols());
  for (int r=0; r < a.rows(); r++) {
    for (int c=0; c < a.cols(); c++) {
      output(r,c) = a(r,c) * b(r,c);
    }
  }
}


int randInt(int lim) {
  return int((double)lim*(double)rand()/(RAND_MAX+1.0));
}

double randDouble(double lim) {
  return (double)lim*(double)rand()/(RAND_MAX+1.0);
}

(in-package :deep-learning)

(declaim (optimize (debug 3)))

(defun train-sparse-autoencoder (nn input-generator &key (max-iterations 4000000))
  "Trains a sparse autoencoder.  INPUT-GENERATOR is a function that
returns.  "
  (dotimes (i max-iterations)
    ;; (i) run a feedforward pass on our network
    (let ((input (funcall input-generator)))
      (forward-propagate-activations nn input)
      (backward-propagate-activations nn input)
      (perform-sparse-learning-pass nn)))
  nn)

(defclass sparse-nn ()
  ((weight-matrices :initform nil :initarg :w :initarg :weight-matrices
		    :accessor nn-weight-matrices
		    :documentation "The parameter associated with the
connection between unit j in layer l, and unit i in layer l + 1.  In
other words, row `unit' in W[layer] corresponds to how much the inputs
from layer `layer - 1' should be weighted going into unit # `unit' in
layer `layer' Thus, row x * activations in x-1 = scalar result.")
   (activations :initform nil :initarg :activations :accessor nn-activations
		:documentation "A sequence of activations vectors.")
   (activation-inputs :initform nil :initarg :activation-inputs
		      :accessor nn-activation-inputs)
   (activation-averages :initform nil :initarg :activation-averages
			:accessor nn-activation-averages)
   (biases :initform nil :initarg :biases
	   :accessor nn-biases
	   :documentation "A sequence of bias vectors.")))

(defun make-sparse-nn (layer-specs)
  "Makes a sparse auto-encoder given a sequence of layer
specifications.  Each layer spec is a number that indicates how many
neurons are in that layer.  Note that we include an input and output
layer in the layer specs sequence.  So for a NN with 1 hidden layer,
there are 3 layer specs."
  (let ((lisp-matrix:*default-implementation* :foreign-array))
    (make-instance 'sparse-nn
		   :activations (map 'vector  #'(lambda (num-units) (make-matrix num-units 1)) layer-specs)
		   :activation-inputs (map 'vector  #'(lambda (num-units) (make-matrix num-units 1)) layer-specs)
		   :activation-averages (map 'vector  #'(lambda (num-units) (make-matrix num-units 1 :initial-element 0)) layer-specs)
		   :weight-matrices (loop :for (lower-num-units num-units) :on layer-specs :until (null num-units)
					  :collect (let* ((m (make-matrix num-units lower-num-units)))
						     (dotimes (row num-units)
						       (dotimes (col lower-num-units)
							 (setf (mref m row col)
							       (coerce (random (/ 1 (sqrt lower-num-units))) 'double-float))))
						     m))
		   :biases (loop :for (lower-num-units num-units) :on layer-specs :until (null num-units)
				 :collect (make-matrix num-units 1 :initial-element .01)))))

(defun nn-layer-count (nn)
  "Returns the number of layers in the neural network, including the
input and output layers."
  (length (nn-activations nn)))

(defun nn-hidden-layer-count (nn)
  "Returns the number of hidden layers in the neural network."
  (- (nn-layer-count nn) 2))

(defun forward-propagate-layer-activations (nn update-layer)
  "Propagates the activations from UPDATE-LAYER - 1 into UPDATE-LAYER."
  (copy!
   (m+ (m* (elt (nn-weight-matrices  nn) (- update-layer 1))
	   (elt (nn-activations  nn) (- update-layer 1)))
       (elt (nn-biases nn) (- update-layer 1)))
   (elt (nn-activations  nn) update-layer)))

(defun forward-propagate-activations (nn input)
  "Propagates the"
  (lisp-matrix:copy! input (elt (nn-activations nn) 0))
  
  (loop :for update-layer :from 1 :upto (1- (nn-layer-count nn))
	:do (forward-propagate-layer-activations nn update-layer))
  (elt (nn-activations nn) (1- (nn-layer-count nn))))

(defparameter *weight-decay* 0.002d0)
(defparameter *alpha* 0.1d0)
(defparameter *beta* 5.0d0)

(defun backward-propagate-activations (nn training-output)
  "so-called `back propagation'"
  (let ((gammas (coerce (loop :for i :from 0 :upto (nn-layer-count nn) :collect nil)
			'vector)))
    (macrolet ((nth-gamma (n) `(elt gammas ,n))
	       (nth-activation (n) `(elt (nn-activations nn) ,n))
	       (nth-bias (n) `(elt (nn-biases nn) ,n))
	       (nth-weight-matrix (n) `(elt (nn-weight-matrices nn) ,n)))
      ;; 1. For the output layer, set gamma = (activation[layer] - trainingOutput) .* activationprime(layer)
      (let* ((last-layer-num (1- (nn-layer-count nn)))
	     (activation-prime (activation-function-derivative nn last-layer-num))
	     (last-layer-gamma
	      (m.* (m- (elt (nn-activations nn) last-layer-num)
		       training-output)
		   activation-prime)))
	(setf (nth-gamma last-layer-num) last-layer-gamma))
      ;; 2. For each other layer, set gamma = ((_W[layer]Transpose * gamma[layer+1]) .* f'(activationFnInputs[layer])
      (loop :for layer-num :from (- (nn-layer-count nn) 2) :downto 1
	    :do  (setf (nth-gamma layer-num)
		       (let ((activation-prime (activation-function-derivative nn layer-num)))
			 (m.* (m* (transpose-matrix (elt (nn-weight-matrices nn) layer-num))
				  (nth-gamma (+ 1 layer-num)))
			      activation-prime))))
      ;; 3. Update parameters according to the gammas
      (loop :for layer-num :from 1 :below (nn-layer-count nn)
	    :do (let* ((A (m* (nth-gamma layer-num)
			      (transpose-matrix (nth-activation (- layer-num 1)))))
		       (B (lisp-matrix:scal *weight-decay* (nth-weight-matrix (- layer-num 1))))
		       (-weight-change (scal *alpha* (m+ A B)))
		       (-bias-change   (scal *alpha* (nth-gamma layer-num))))
		  (copy! (m- (nth-weight-matrix (- layer-num 1)) -weight-change)
			 (nth-weight-matrix (- layer-num 1)))
		  (copy! (m- (nth-bias (- layer-num 1)) -bias-change)
			 (nth-bias (- layer-num 1))))))))

(defparameter *target-activation* .04d0)
(defparameter *running-average-decay* .999d0)

(defun perform-sparse-learning-pass (nn)
  "Update the weights and the like such that they trend towards a
sparse activation pattern. This is the third step in each learning
iteration."
  (macrolet ((nth-activation (n) `(elt (nn-activations nn) ,n))
	     (nth-bias (n) `(elt (nn-biases nn) ,n))
	     (nth-weight-matrix (n) `(elt (nn-weight-matrices nn) ,n)))
    (loop :for layer-num :from 1 :below (nn-layer-count nn)
	  :do (progn
		;; update the activation values for this layer to approach average activation
		(copy! (m+ (m* (nth-weight-matrix (- layer-num 1))
			       (nth-activation (- layer-num 1)))
			   (nth-bias (- layer-num 1)))
		       (elt (nn-activation-inputs nn) layer-num))
		(copy! (elt (nn-activation-inputs nn) layer-num)
		       (nth-activation layer-num))
		;; update average activations for this layer
		(let ((average-activations
		       (m+ (scal (- 1 *running-average-decay*) (nth-activation layer-num))
			   (scal *running-average-decay*       (elt (nn-activation-averages nn) layer-num)))))
		  (copy! average-activations (elt (nn-activation-averages nn) layer-num))
		  ;; update the biases
		  (let ((bias (nth-bias (- layer-num 1))))
		    (loop :for unit :from 0 :below (nn-layer-count nn)
			  :do (let ((bias-adjustment (* -1.0 *alpha* *beta*
							(- (vref average-activations unit) *target-activation*))))
				(incf (vref bias unit) bias-adjustment)))))))))
		    
(defun activation-function-derivative (nn layer-num)
  "Returns a matrix with the derivative of the activation function
applied to layer LAYER-NUM in the sparse neural network NN."
  (let ((activation-vector (elt (nn-activations nn) layer-num)))
    (mapmat-into (copy activation-vector)
		 #'(lambda (activation) (- 1 (* activation activation)))
		 activation-vector)))
	       
(defun layer-size (nn layer-num)
  "Returns the dimension of layer number LAYER-NUM in the neural
network NN. "
  (length (elt (nn-activations nn) layer-num)))

(defun mapmat-into (into fn &rest matrices)
  "Maps over each element of the matrices and sets the corresponding
element in INTO to the result of applying FN to the , and sets the result "
  (let ((rows (nrows into))
	(cols (ncols into)))
    (assert (apply #'= rows (map 'list #'nrows matrices)))
    (assert (apply #'= cols (map 'list #'ncols matrices)))
    (loop :for row :from 0 :upto (1- rows)
	  :do (loop :for col :from 0 :upto (1- cols)
		    :do (setf (mref into row col)
			      (apply fn (map 'list #'(lambda (m) (mref m row col)) matrices))))))
  into)

  

(defgeneric activation-fn (nn input output)
  (:documentation "Activation function.  Applies some sigmoid like
function to the input matrix, and stores the result in output."))

(defmethod activation-fn ((nn sparse-nn) (input matrix-like) (output matrix-like))
  (mapmat-into output #'tanh input))



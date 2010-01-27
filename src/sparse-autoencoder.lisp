(in-package :deep-learning)

(defclass sparse-nn ()
  ((weight-matrices :initform nil :initarg :w :initarg :weight-matrices
		    :accessor nn-weight-matrices
		    :documentation "The parameter associated with the
connection between unit j in layer l, and unit i in layer l + 1.  In
other words, row `unit' in W[layer] corresponds to how much the
inputs from layer `layer - 1' should be weighted going into unit #
`unit' in layer `layer' Thus, row x * activations in x-1 = scalar
result.")
   (activations :initform nil :initarg :activations :accessor nn-activations
		:documentation "A sequence of activations vectors.")
   (activation-inputs :initform nil :initarg :activation-inputs
		      :accessor nn-activation-inputs)
   (activation-averages :initform nil :initarg :activation-averages
			:accessor nn-activation-averages)
   (biases :initform nil :initarg :biases
	   :accessor nn-biases
	   :documentation "An sequence of bias vectors.")))

(defun make-sparse-nn (layer-specs)
  (let ((lisp-matrix:*default-implementation* :foreign))
    (make-instance 'sparse-nn
		   :activations (map 'array  #'(lambda (num-units) (make-matrix num-units 1)) layer-specs)
		   :activation-inputs (map 'array  #'(lambda (num-units) (make-matrix num-units 1)) layer-specs)
		   :activation-averages (map 'array  #'(lambda (num-units) (make-matrix num-units 1 :initial-element 0)) layer-specs)
		   :weight-matrices (loop :for (lower-num-units num-units) :on layer-specs :until (null num-units)
					  :collect (let ((m (make-matrix lower-num-units num-units)))
						     (dotimes (row lower-num-units)
						       (dotimes (col num-units)
							 (setf (mref m row col) (random (/ 1 (sqrt lower-num-units))))))
						     m))
		   :biases (loop :for (lower-num-units num-units) :on layer-specs :until (null num-units)
				 :collect (make-matrix lower-num-units 1 :initial-element 0)))))

(defun nn-layer-count (nn)
  (length (nn-activations nn)))

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
	:do (forward-propagate-layer-activations nn update-layer)))


(defun backward-propagate-activations (nn input)
  (loop :for update-layer :from 1 :upto (1- (nn-layer-count nn))
	:do (forward-propagate-layer-activations nn update-layer)

  
  
(defun train-sparse-autoencoder (nn input-generator &key (max-iterations 4000000))
  "Trains a sparse autoencoder.  INPUT-GENERATOR is a function"
  
  (dotimes (i max-iterations)
    ;;    // (i) run a feedforward pass on our network
    (let ((input (funcall input-generator)))
      (forward-propagate-activations nn input)
      (backward-propagate-activations nn input)
      (sparse-learning-pass nn)))
  nn)

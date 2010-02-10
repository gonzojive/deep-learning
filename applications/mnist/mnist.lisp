(require :ironclad)
(require :deep-learning)

(defpackage :mnist
    (:use :common-lisp :lisp-matrix))

(in-package :mnist)

(declaim (optimize (debug 3)))

;;; utility
(defun make-simple-array (num-bytes)
  (make-array num-bytes :element-type '(unsigned-byte 8)))

(defun read-image-file (file)
  (let ((image-matrices nil))
    (with-open-file (s file :element-type '(unsigned-byte 8))
      (flet ((read-int ()
	       (let ((seq (make-simple-array 4)))
		 (unless (= 4 (read-sequence seq s))
		   (error "Invalid integer read on MNIST image file."))
		 (ironclad:octets-to-integer seq))))
	(let ((magic-number (read-int))
	      (num-images (read-int))
	      (num-rows (read-int))
	      (num-columns (read-int)))
	  (assert (= 2051 magic-number))
	  (flet ((read-image ()
		   (let ((matrix (make-matrix num-rows num-columns))
			 (image-bytes (let ((seq (make-simple-array (* num-rows num-columns))))
					(read-sequence seq s)
					seq)))
		     (dotimes (row num-rows)
		       (dotimes (col num-columns)
			 (let ((byte (elt image-bytes (+ (* row num-columns) col))))
			   (setf (mref matrix row col)
				 (coerce (- (* 2.0d0 (* byte (/ 255))) 1.0d0) 'double-float)))))
		     matrix)))
	    ;;(let ((row (make-simple-array num-columns)))
	    ;;(read-sequence row s)
	    ;;(format t "|~A|~%" (map 'string #'(lambda (x) (if (< 100 x) #\X #\ ))   row))))))
	    (dotimes (i num-images)
	      (push (read-image) image-matrices))))))
    (coerce (nreverse image-matrices) 'vector)))


(defvar *training-images* nil
  )

(defun one-dimensionalize (matrix)
  (let ((result (make-matrix (* (nrows matrix) (ncols matrix)) 1)))
    (dotimes (row (nrows matrix))
      (dotimes (col (ncols matrix))
	(setf (mref result (+ col (* row (ncols matrix))) 0)
	      (mref matrix row col))))
    result))

(defvar *training-inputs*
  (map 'vector 'one-dimensionalize
       (mnist::read-image-file "/git/deep-learning/applications/mnist/data/train-images-idx3-ubyte")))

(defun train-sparse-digit-classifier ()
  (let ((nn (deep-learning:make-sparse-nn
	     (list (* (nrows (elt *training-inputs* 0)) (ncols (elt *training-inputs* 0)))
		   120
		   (* (nrows (elt *training-inputs* 0)) (ncols (elt *training-inputs* 0)))))))
    (deep-learning:train-sparse-autoencoder nn
					    #'(lambda () (elt *training-inputs* (random (length *training-inputs*))))
					    :max-iterations 100)
    nn))
		      
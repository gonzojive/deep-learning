;;;; jwacs.asd
;;;
;;; This is the system definition file for the jwacs project.
;;; It defines the asdf system plus any extra asdf operations
;;; (eg test-op).

(defpackage :deep-learning-system
  (:use :cl :asdf)
  (:export
   #:*version*
   #:*executable-name*))

(in-package :deep-learning-system)

;;;; ======= System definition =====================================================================
(asdf:defsystem deep-learning
  :version *version*
  :author "Red Daly"
  :licence "MIT License <http://www.opensource.org/licenses/mit-license.php>"
  :serial t
  :components ((:module
		"src"
		:components
		((:file "package")
		 (:file "sparse-autoencoder" :depends-on ("package"))
		 )))

  :depends-on (:alexandria :anaphora :lisp-matrix))

#+nil
(defsystem deep-learning-tests
  :components ((:module "test"
                        :components ((:file "test-package")
				     (:file "memory-tests" :depends-on ("test-package"))
				     )))
  :depends-on ("deep-learning" "stefil"))

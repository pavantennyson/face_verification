# Face verification


## Project Description

This is a face verification project which is similar to face unlock in mobile and other devices,<Br>
  so first the face is registered through webcam and then later registered face is used to verify whether the face is matching or not
  
## Files and Usage

* face_registration.py
  * This file will lauch webcam for face registration
  * Press "S" to register face 
  * To quit without registering press "Q"
* face_verification.py
  * This will launch the webcam to verify the face (press "Q" to exit)
 * face_tools.py - contains code required to extract and predict face
 
 ## Required libraries
 * [MTCNN](https://pypi.org/project/mtcnn/0.1.0/)
 * [OpenCV](https://pypi.org/project/opencv-python/)
 * [keras-vggface](https://github.com/rcmalli/keras-vggface)
 
 ## References
 
 [How to Perform Face Recognition With VGGFace2 in Keras](https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/)
 [OpenCV-Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
 [keras-vggface](https://github.com/rcmalli/keras-vggface)

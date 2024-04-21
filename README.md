# Driver_drowsiness_system_CNN

This is a system which can detect the drowsiness of the driver using CNN - Python, OpenCV

The aim of this is system to reduce the number of accidents on the road by detecting the drowsiness of the driver and warning them using an alarm. 

It detects wheather the person is drowsy , yawning , or not looking forward , alert with sound send to the driver.

Here, we used Python, OpenCV, Keras(tensorflow) to build a system that can detect features from the face of the drivers and alert them if ever they fall asleep while while driving. The system dectects the eyes and prompts if it is closed or open. If the eyes are closed for 3 seconds it will play the alarm to get the driver's attention, to stop cause its drowsy.We have build a CNN network which is trained on a dataset which can detect closed and open eyes. Then OpenCV is used to get the live fed from the camera and run that frame through the CNN model to process it and classify wheather it opened or closed eyes.

## Setup
To set the model up:<br />
Pre-install all the required libraries <br />1) OpenCV<br />
                                       2) Keras<br />
                                       3) Numpy<br />
                                       4) Pandas<br />
                                       5) OS<br />
                                       6) Playsound<br />
                                       7) dlib<br />
                                       8) streamlit
                                       9) pygame


Download the Dataset from the link given below and edit the address in the notebook accordingly.<br />
Run the Jupyter Notebook and add the model name in detect_drowsiness.py file in line.<br />

## The Dataset
The dataset which was used is a subnet of a dataset from(https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)<br />
it has 4 folder which are <br />1) Closed_eyes - having 726 pictures<br />
                          2) Open_eyes - having 726 pictures<br />
                          3) Yawn - having 725 pictures<br />
                          4) no_yawn - having 723 pictures<br />


Download DLib library , and 68_facial_lankmarks_dat file.

Arrange the directory structure accordingly .

1. Create a virtual environment and activate it.

2. then go to client folder and install the requirements 

	-> pip install -r requirements.txt 

3. pose_utils.py is used to calculate the 24 features mentioned in the file. This features are used to record exercises. And the model is trained on this features.

4. You can use camera_test.py to test the demo to see the 24 features.
 	-> python camera_test.py

5. auto_recorder.py is used to record the data by your own or by providing a path to video file.
	
   you can record any of the six exercises provided in the list by running this command :

	-> python auto_recorder.py

6. Once the exercises are recorded, run the build_exercise_dataset.py to combine the recorded data.

	-> python build_exercise_dataset.py

7.After collecting the data run train_exercise_model.py to train the LSTM model
	
 	-> python train_exercise_model.py

8. Once the model is trained you can test it by running :

	-> python exercise_demo.py

9. Perform one of the six exercise in front of the camera and you will see the results on the screen.

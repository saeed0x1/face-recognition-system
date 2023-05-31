# face-recognition-system

Face recognition attendance system built in python.

### USED MODULES
```sh
- open-cv
- numpy
- os
- datetime
- face_recognition
```

### SET UP

**Install the required modules**
```sh
pip install -r requirements.txt
```
### Run 
Before running the program put all the images with correct names in the `images` directory.

```
python main.py
```
After running the program it'll open up your computer's webcam and when you put the images in front of the camera and if it exists in the images directory it'll show the name and record it inside `attendance.csv` with name and timestamp. 

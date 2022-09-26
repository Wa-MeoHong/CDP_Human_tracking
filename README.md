# CDP_Human_tracking
CDP_Human_tracking is an Autonomous driving system that follows a person 
implemented in Python.

Used pip module : tensorflow, tflite, tflite_runtime, opencv-python, numpy, pyvesc, pythoncrc, pyserial
Used Devices : Raspberry PI, Servomotor, BLDCMotor, VESC, Google-Coral USB Accelerator

tracking_master.py 
  A file that causes the autonomous driving module to run when the Raspberry Pi is booted.
  If you want to run the module when the Raspberry Pi boots, you can create an sh file (shell script) and run it when booting. 
  
Human_tracking.py
  The core file containing the autonomous driving code, called through tracking_master.py
 
common2.py
  File to load edgeTPU TensorFlow model
  

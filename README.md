# PyQt5-Realsense-Data-Acquisition
a pyqt5 program for realsense to collect depth,rgb,3d points

module python=3.7.16 pyqt5=5.15.9 pyqt5-sip=12.12.2 pyrealsense2=2.54.1.5216 numpy=1.21.6 opencv-python=4.8.0.76 pyuic pyinstaller pyqt5-tools

.ui -> .py 
pyuic5 -o *.py *.ui

for package
pyinstaller -D -w *.py
Tips:the numpy folder maybe error,use the anaconda->envs->packageSite->numpy to rewrite it!

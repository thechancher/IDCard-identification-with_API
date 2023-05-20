# IDCard-identification-with-API

This project is developed and tested in the  [Jetson Nano, JetPack 4.6.3 with Jetson Linux 32.7.3](https://developer.nvidia.com/jetpack-sdk-463).

![JetPack-1](/docs/JetPack-1.png)

![JetPack-2](/docs/JetPack-2.png)

This project is the improvement of the [previous version](https://github.com/Maesly/IDCard-identification-for-an-embeddec-system), 
which separates the database, and places it in the cloud. This allows the embed to focus only on decoding the image and obtaining the necessary data.

Firstly, for the installation of the project the version of Python 3.7 and the pip version 3 are required.

## Install Python:
Follow these steps: https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/

## Install pip
Follow these steps: https://www.odoo.com/es_ES/forum/ayuda-1/how-to-install-pip-in-python-3-on-ubuntu-18-04-167715

## Install numpy
sudo apt-get install python-numpy
en caso de tener problemas con la versión, realizar una purga:
```
sudo -H python3.7 -m pip uninstall numpy
sudo apt purge python3-numpy -y
sudo -H python3.7 -m pip install --upgrade pip
sudo -H python3.7 -m pip install numpy
```

## Install opencv
```
sudo pip3 install opencv-python
```

## Install matplot
```
sudo apt-get install python3-matplotlib -y
```

## Install pandas
```
sudo apt-get install python3-pandas -y
```

## Install tesseract
```
sudo pip3 install tesseract
sudo apt-get install tesseract-ocr -y
sudo apt-get install tesseract-ocr-spa
sudo pip3 install tesserocr
```

## Install others
```
sudo apt-get install libtesseract-dev -y
sudo apt-get install libleptonica-dev
sudo apt-get install pkg-config
sudo pip3 install Levenshtein
sudo pip3 install fuzzywuzzy
```

## Web Cam
to verify that the camera is recognized it has to be listed with the command:

```
ls /dev/video*
```

![video0](/docs/video0.png)

this name is assigned in the parameter on line 8 of the file:

 **main.py**

so far, for some other documentation, go to:

[IDCard-identification-for-an-embeddec-system](https://github.com/Maesly/IDCard-identification-for-an-embeddec-system)

## API
to change the key code to access the API, refer to the API class parameter, which is in the file:

**API.py** 

the images that the camera captures are saved in the folder:

**id-validate**

In case of testing without a webcam, comment the lines 8 and 9 of the file: **main.py**, save the images you want to process in the same folder, and the files with a size of 640x400, with extension .jpg




# Face-Recognition

### 目標 : 使用TX2設備，偵測有多少人進入過畫面並統計停留時間，也判斷其性別與臉部表情。

### 詳細教學簡報:
* [NVIDIA Jetson TX2之人臉辨識](https://drive.google.com/open?id=1DAThAOme3e5eERBfAnlg8tOm-HSmw6BO)

### 使用方法:
1. 安裝套件 : 
* 在windows上執行 `pip3 install -r requirements.txt` 
* 在TX2上請依照簡報教學安裝
2. 下載模組 :
* [model](https://drive.google.com/open?id=1_woLikMLfRrE85wUdJ4yRX4c7-BNTqHB)
3. 執行程式 : 
* `python3 camera_face_recognition.py TX2` (TX2 改 WIN 在 windows下執行)

##### 顯示特徵點程式:
* `python3 landmarks_demo.py`

### 參考資料:
* https://github.com/davisking/dlib
* https://github.com/ageitgey/face_recognition
* https://github.com/oarriaga/face_classification

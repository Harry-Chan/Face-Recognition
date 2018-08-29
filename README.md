# Face-Recognition

### 目標 : 在一段時間內偵測有多少人進入過畫面並統計停留時間，也判斷其性別與臉部表情。

### 詳細教學簡報:
* [NVIDIA Jetson TX2之人臉辨識](https://drive.google.com/open?id=1rNVhzhzan2oB_g_e7ap3UiJ1vEmKkWl_)

### 使用方法:
1. 安裝套件 : 
* 在windows上執行 `pip3 install -r requirements.txt` 
* 在TX2上請依照簡報教學安裝
2. 下載模組 :
* [model](https://drive.google.com/file/d/1G8vmmQxbwtGRhnIPTkmHDpitthfDuLoQ/view)
3. 執行程式 : 
* python3 camera_face_recognition.py TX2 (TX2 改 WIN 在 windows下執行)
4. 顯示特徵點程式:
* python3 landmarks_demo.py

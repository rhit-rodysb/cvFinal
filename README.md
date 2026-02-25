# Multi-Object Tracking with YOLOv11 on MOT17 Dataset
This project runs MOT on the MOT17 benchmark using YOLOv11 and ByteTrack.

---

## Setup

### Dependencies
Create a virtual environment then install all required packages:
```bash
python -m venv yoloenv # you can change the name "yoloenv"
source yoloenv/bin/activate #Linux/Mac
yoloenv\Scripts\activate #Windows
```
Once in the virtual environment:
```bash
pip install ultralytics
pip install motmetrics
pip install openv-python
pip install numpy
```

---

### Dataset
Download MOT17 from https://motchallenge.net and extract it so the structure looks like this:
```
./
  MOT17/
    train/
      MOT17-02-DPM/
        img1/
        gt/
          gt.txt
        seqinfo.ini
      MOT17-04-DPM/
      ...
```

---

### ffmpeg

The code uses ffmpeg to compile the individual frames into mp4 videos. Download a static ffmpeg binary and place it at `~/ffmpeg`:

https://ffmpeg.org/download.html

Or, on Linux, you can do:
```bash
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg-release-amd64-static.tar.xz
cp ffmpeg-*/ffmpeg ~/ffmpeg
chmod +x ~/ffmpeg
```
---

### Model weights
`yolo11n.pt` will auto-download when you first run the script.

---

## Running the Code
The code auto-selects the GPU(s) with the least usage.

To run using 1 GPU:
```bash
python runSerial.py
```

To run using 3 GPUs:
```bash
python runParallel.py
```

Results will be saved to `./results/`

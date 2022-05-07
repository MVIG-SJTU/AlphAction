# Demo

### Installation 

To run this demo, make sure that you install all requirements following [INSTALL.md](../INSTALL.md).

### Preparation

1. Download the object detection model manually: **yolov3-spp.weights** ([Google Drive](https://drive.google.com/file/d/1260DRQM5XtSF7W213AWxk6RX2zfa3Zq6/view?usp=sharing)). Place it into `data/models/detector_models`.
2. Download the person tracking model manually: **jde.uncertainty.pt** ([Google Drive](https://drive.google.com/file/d/1nuCX5bR-1-HGZ0_WoH4xZzPiV_jgBphC/view?usp=sharing)). Place it into `data/models/detector_models`.
3. Please download our action models. Place them into ```data/models/aia_models```. All models are available in [MODEL_ZOO.md](../MODEL_ZOO.md).
4. We also provide a practical model ([Google Drive](https://drive.google.com/file/d/1gi6oKLj3wBGCOwwIiI9L4mS8pznFj7L1/view?usp=sharing)) trained on 15 common action categories in AVA. This 
model achieves about 70 mAP on these categories. Please use [`resnet101_8x8f_denseserial.yaml`](../config_files/resnet101_8x8f_denseserial.yaml)
and eable `--common-cate` to apply this model.

### Usage

1. Video Input

    ```
    cd demo
    python demo.py --video-path path/to/your/video --output-path path/to/the/output \ 
    --cfg-path path/to/cfg/file --weight-path path/to/the/weight [--common-cate] 
    ```

2. Webcam Input

    ```
    cd demo
    python demo.py --webcam --output-path path/to/the/output \
    --cfg-path path/to/cfg/file --weight-path path/to/the/weight [--common-cate] 
    ```

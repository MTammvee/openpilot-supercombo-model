# Openpilot supercomo model deployment

Using comma.ai pretrained self-driving car model to predict lane lanes. 
![comma_exmpl](https://user-images.githubusercontent.com/43088163/118095593-f0408480-b3d8-11eb-8837-c4cd3f59eed6.png)



# Installation

Use Python version >= 3.6 
1. Install requirements
```sh
$ pip3 install -r requirements.txt
```
2. Use your own video or download my sample video from [HERE](https://drive.google.com/file/d/10CFyMSEY_w5ZjzWsYClFxYIdpY62PG31/view?usp=sharing).
```sh
$ mkdir data
$ cd data
```
3. Download pre-trained model (onnx) from [comma-ai gituhub](https://github.com/commaai/openpilot/tree/master/models)

4. Run the program

Note: Specify the video feed location
```sh
$ python3 openpilot_onnx.py
```

### Credits

Thank You comma.ai for making your research open source.

| Credit | Link |
| ------ | ------ |
| Comma Ai | [https://comma.ai/] |
| GitHub | [https://github.com/commaai] |
| Trained models | [https://github.com/commaai/openpilot/tree/master/models] |



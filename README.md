# Openpilot supercomo model deployment

Using comma.ai pretrained self-driving car model to predict lane lines.
![output](https://user-images.githubusercontent.com/43088163/120663559-cab41180-c492-11eb-940d-c58e9b5983f7.png)




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



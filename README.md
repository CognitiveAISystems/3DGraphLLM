# 3DGraphLLM

In this work, we propose 3DGraphLLM, a method for constructing a learnable representation of a 3D scene graph, which serves as input for LLMs to perform 3D vision-language tasks.

## News

[2024.12] We release 3DGraphLLM [paper] [code]

## 🔨 Preparation

- Prepare the environment:
  
  ```shell
  conda create -n 3dgraphllm python=3.9.17
  conda activate 3dgraphllm
  conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install -r requirements.txt
  ```
- If you don't have root permissions to install java (needed for pycocoeval scripts for metrics such as BLEU and CIDER), install it with conda:

```
conda install -c conda-forge openjdk
```

  
- Download LLM backbone:
  -  We use LLAMA3-8B-Instruct in our experiments, which can be downloaded from [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

  - Change the `llama_model_path` in [config.py](./scripts/config.py) to the path of `LLAMA3-8B-Instruct`.
  

- Annotations and extracted features:
  
  Please follow the instructions in [preprocess](preprocess/).


## 🤖 Training and Inference

- Training
  - Modify [run.sh](scripts/run.sh):
    ```python
    train_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref#nr3d_caption#obj_align"
    val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
    evaluate=False
    ```

    <details>
    <summary> Explanation of "train_tag" and "val_tag" </summary>

    - Use `#` to seperate different datasets

    - Datasets:
      - `scanrefer`: [ScanRefer](https://github.com/daveredrum/ScanRefer) Dataset
      - `scan2cap`: [Scan2Cap](https://github.com/daveredrum/Scan2Cap) Dataset
      - `scanqa`: [ScanQA](https://github.com/ATR-DBI/ScanQA) Dataset
      - `sqa3d`: [SQA3D](https://github.com/SilongYong/SQA3D) Dataset
      - `multi3dref`: [Multi3dRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP) Dataset
      - `nr3d_caption`: A captioning dataset originated from [Nr3D](https://github.com/referit3d/referit3d).
      - `obj_align`: A dataset originated from ScanRefer to align the object identifiers with object tokens.

    </details>
  - Run: `bash scripts/run.sh`


- Inference
  
  - Modify [run.sh](scripts/run.sh): We provide the pretrained checkpoint in [Yandex Disk](https://disk.yandex.ru/d/fTz0uzDNHjI4mw)
  
    ```python
    val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
    evaluate=True
    pretrained_path="/path/to/pretrained_model.pth"
    ```
  
  - Run: `bash scripts/run.sh`
  
## 🚀 Demo

- Run: `bash demo/run_demo.sh`. You will be prompted to ask different queries about Scene 435 of ScanNet.


## 📪 Contact

If you have any questions about the project, please open an issue in this repository or send an email to [Tatiana Zemskova](zemskova@airi.net).

## 📑 Citation

If you find this work helpful, please consider citing our work as:



## 😊 Acknowledgement

Thanks to the open source of the following projects:

[Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene/tree/dev)

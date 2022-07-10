1. Generate Datasets

```bash
python preprocess.py 
```

2. Unit Test

```bash
cd unit_test && pytest rawdata_test.py -s
```

3. Log

```bash
tensorboard --logdir=lightning_logs/ --bind_all --port 6006
```

4. Move data

```
mkdir pycorrector/pycorrector/macbert/output/
cp -r *.json /data/zhangruochi/projects/AuditCorrect/pycorrector/pycorrector/macbert/output/
```

5. Download pretraind model
```
git lfs install
git clone https://huggingface.co/shibing624/macbert4csc-base-chinese
```

6. Training
```
cd pycorrector/pycorrector/macbert && python train.py
```

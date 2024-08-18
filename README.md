# Environment construction
```bash
conda create -n signal python pip
conda activate signal
pip install -r req.txt
```

# Prepare Data
### [capture-24 dataset](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001)
```bash
wget -O capture24.zip https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b
unzip capture24.zip
```
```bash
python raw2npy.py
```

### dataset for downstream (ADL, Opportunity, PAMAP2, REALWORLD, WISDM) 
```bash
wget -O downstream.zip https://www.dropbox.com/scl/fi/a9x97npde4qdazv1lxlyu/downstream.zip?rlkey=y3xv06fxh2lv47m0zoy1f310q&st=bo2l4yhv&dl=0
unzip downstream.zip
mv downstream ./data/
```

# Pre-Training
### representation condition
We train the Representation-conditional Diffusion Transformer, which are given by the output of a transformer.
```bash
python3 pre_train.py
```

### Results
after the above code is executed, the results are output to `experiment_log/pre-train/`.

# downstream
### RepcondFormer
```bash
python3 downstream.py \
  --ckpt <pre-trained model path> \
  --dataset adl \
```

### RepcondClassifier
##### tuning
```bash
python3 downstream_train.py \
  --ckpt <pre-trained model path> \
  --dataset oppo \
```
##### zero-shot classify
```bash
python3 downstream_zeroshot_classify.py \
  --ckpt <fine-tuned model path>
```

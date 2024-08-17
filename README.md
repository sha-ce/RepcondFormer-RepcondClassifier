# Data Parsing
`.csv.bz2`ファイルから time window に分割したデータへ変換し`.npy`ファイルとして保存します．
```bash
python3 raw2npy.py
```
デフォルトで実行すると，`../signal_data/`ディレクトリ内のすべての`csv.bz2`ファイルを読み込み，`.npy`ファイルを`./data/`内に保存します．<br>
デフォルトでは，`./data/50hz_4.0s_overlap2.0s/train.npy`, `/test.npy`が保存されます．

# Pre-Training
### uncondition
Signal版の DiTアーキテクチャを使用して，訓練します．
```bash
python3 pre_train.py
```
無条件の拡散モデルでモデリングされた学習が行われます．

### representation condition
表現条件付きの拡散モデルの訓練をします．表現はTransformerの出力で得られます．
```bash
python3 pre_train.py --rep-condition
```

### Results
after the above code is executed, the result is output to `experiment_log/pre-train/`.

# downstream
### classify using transformer
Diffusionモデルに内に組み込んだ表現にエンコードするTransformerのみを使用して分類を行います．`--ckpt`に事前学習時のモデルのパスを指定してください．`--dataset`にデータセットを指定できます．`adl`，`oppo`，`pamap`，`realworld`，`wisdm`の5つから選択してください．`--transfer`をつけると，転移学習を行います．付けずに実行するとファインチューニングになります．他にも実行時に指定できるパラメータがあります．詳細は[`downstream.py`](./downstream.py)を参照してください．
```bash
python3 downstream.py \
  --ckpt <pre-trained model path> \
  --dataset oppo \
  --transfer
```

### fine-tuning diffusion model for classify
##### training
分類を行うための拡散モデルを構築するためにデータセットの訓練データを使用しファインチューニングをを行います．
```bash
python3 downstream_train.py \
  --ckpt <pre-trained model path> \
  --dataset oppo \
```
##### zero-shot classify
ファインチューニングを行ったモデルのパスを`--ckpt`に指定し，暗黙的にクラス分類子を取り出し，分類を行います．
他のパラメータに関して，<br>
- `--seed`はシード値を表します．
- `--denoise-num`はタイムステップ`--timestep`で指定した値から連続でデノイズする回数を表します．デフォルトは1でこの値を増やしてもほとんど性能は変化しなかったことを実験で確認しました．
- `--num-per-class`はクラスごとのサンプリングする数を表します．大きければ大きいほど大数の法則からシード値に左右されない精度が期待できます．
- `--timestep`はデノイズするタイムステップを表します．元画像を予測したものと元画像との平均二乗誤差の大きさを評価しているので，`500`以上はあまりいい性能になることはないと思われます．
```bash
python3 downstream_zeroshot_classify.py \
  --ckpt <fine-tuned model path>
```


# Sampling
次のコマンドを実行することで信号のサンプリングをします．
`--ckpt`に学習済みモデルパラメータのパスを指定してください．
`--num-samples`にサンプル数を指定できます．デフォルトは`50,000`です．<br>
50,000の信号を生成するため時間がかかります．処理が終わると`.npz`ファイルが保存されます．評価の時にこのファイルを使用します．
```bash
python3 sample.py
```

# Evaluation
二つのデータセットの`.npz`ファイルを指定して実行します．
```bash
cd evaluate
python3 evaluate.py *original-dataset-npzfile-path *sample-dataset-npzfile-path
```

### result

以下の4つの指標が返ってきます．
```bash
FID:  ***
Inception Score:  ***
Precision: ***
Recall: ***
```
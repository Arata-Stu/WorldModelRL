# WorldModel RL
このレポジトリはVAEを用いた強化学習を行う

## Setup
ubuntu 22.04でテストを動作を確認済み

``` shell
# 例 python3.11 をインストール
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt -y update
sudo apt install -y python3.11
sudo apt install -y python3.11-venv
```

```shell
# python3.11 -m venv <env_name>
# source <env_name>/bin/activate

#例 gym_envという名前の仮想環境を作成
python3.11 -m venv gym_env
source gym_env/bin/activate
```

## 2. VAEを学習させる

### 2-1. スクラッチからCOCOで学習
```shell
python3 train_vae.py \
data=coco \
model/vae=cnn \
save_ckpt_dir=./ckpts/coco_cnn_vae/ \
data.data_dir=./datasets/coco/ 
data.num_workers=10 
data.batch_size=64
data.img_size=64
```

### 2-2. gym環境のデータを用いてVAEを学習

### データを集める
```shell
python3 collect_data.py \
envs=car_racing \
output_dir=./datasets/car-racing/ \
envs.img_size=64 \
num_episodes=100 \
num_steps=1000 \
num_workers=10 
```

### 集めたデータで学習
```shell
python3 train_vae.py \
model/vae=cnn \
data=gym_img \
save_ckpt_dir=./ckpts/car-racing_cnn/ \
data.data_dir=./datasets/car-racing/ \
data.num_workers=10 
data.batch_size=64
data.img_size=64
```

## 3. MDN RNNを学習させる

### データをエンコード
```shell
python3 encode_dataset.py \
model/vae=cnn \
data_dir=./datasets/car-racing \
output_dir=./datasets/car-racing 
model.vae.ckpt_path=<ckpt>
```

### 学習
```shell
python3 train_mdn.py \
data=mdn \
model/mdn=mdnrnn \
data.data_dir=./datasets/car-racing/ \
save_ckpt_dir=./ckpts/car-racing_mdn/
```



### 3. 強化学習
```shell
python3 train_actor.py vae=cnn buffer=off_policy agent=sac \
vae.ckpt_path=./ckpts/<ckpt_path> \
save_ckpt_dir=./ckpts/<ckpt_dir_name> \
envs.render_mode=rgb_array
```
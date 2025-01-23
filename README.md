# ImageClassificationApp
A serverless web application for classifying images

## Quickstart

Git clone our project:

```shell
git clone https://github.com/mdalek/ImageClassificationApp.git
cd ImageClassificationApp
```


Create a virtual enviroment:

```shell
python -m venv .venv
```

Activate it:

```shell
.venv\scripts\activate
```

Install python dependencies:

```shell
pip install - r requirements.txt
```

## Run

### Train a model

To train a model run:

```shell
python scripts/train.py
```

### Classify an image

```shell
python scripts/classify.py  <path/to/your/image.png>
```

### Continue training an existing model

```shell
python scripts/continue_training.py
```


Run test script:
```shell
python scripts/test.py
```
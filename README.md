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

## Scripts

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

## Frontend

Make sure to change directory into the `frontend/`:

```shell
cd frontend/
```

To install the frondend dependecies run:

```shell
npm install
```

To run the development server run:

```shell
npm run start
```

And to build the production files:

```shell
npm run build
```
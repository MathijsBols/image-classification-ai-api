# image-classification-ai-api
Simple script in python to classify images from the cifar-10 dataset. Uses an post-api.


## Demo

[Demo](https://ai.mathijsbols.nl)


## Features

- Simple web ui
- Image Classification with 10 labels
- Runs local
- Able to make and deploy your own neural network

## Run Locally

Clone the project

```bash
  git clone https://github.com/MathijsBols/image-classification-ai-api
```

Go to the project directory

```bash
  cd image-classification-ai-api
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python api.py
```

Open WebUI

[127.0.0.1:5000](http://127.0.0.1:5000)

## Run in docker

Pull the image

```bash
  docker pull mathijsbols/image-classification-ai-api
```

Run docker container

```bash
  docker run -d -p 5000:5000 mathijsbols/image-classification-ai-api
```

Open WebUI

[127.0.0.1:5000](http://127.0.0.1:5000)
## Authors

- [@MathijsBols](https://github.com/MathijsBols)


## Feedback

If you have any feedback, please reach out to us at info@mathijsbols.nl


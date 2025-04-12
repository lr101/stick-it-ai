# Stick-It AI

This repo contains scripts to extract data from the stick-it backend server, train a reid classification model, as well as a simple python backend to expose a classification api.

Uses torchreid with a pretrained resnet50 model to train on images from the stick-it app for classification of groups depending on the image.

The model is supposed to be used in the stick-it app to automatically detect which groups a sticker belongs to.
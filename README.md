# sample-hunter

Sample Hunter is a system to identify samples in songs with machine learning. A convolutional neural network generates meaningful embeddings of audio, which can then be used in a sequence similarity search with a vector database. This repository contains the code to train the neural network and use it for inference (in the pipeline directory) as well as the code to gather and collate data. With the current configuration, the network can be trained on the Google Colab T4 GPU runtime.

Acknowledgements

* Audio data was scraped from Hip Hop is Read: https://www.hiphopisread.com/
* This project was inspired by Google's Hum to Search system: https://research.google/blog/the-machine-learning-behind-hum-to-search/

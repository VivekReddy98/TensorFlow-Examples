<a id="top"></a>
# TFRecords Tutorial

**Contents**<br>
[What are TFRecords](#what-are-tfrecords)<br>
[Use Case: DeepFake Detection](#deepfake-detection)
[Creating TFRecords](#where-to-put-it)<br>
[Feeding TFRecords for Model Training using tf.data](#writing-tests)<br>


## What are TFRecords

TF Records are Tensorflow's recommended and own binary storage format. When the training datasets are large, using TF Records would lead to a much better performance in feeding data to the model and thus leading to shorter training times. More about this on [Tensorflow.org](https://www.tensorflow.org/tutorials/load_data/tfrecord)

References to Well-written articles on TFRecords:
1. http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
2. https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564

## DeepFake-Detection

The code snippets in the next sections are taken from my other project related to DeepFake Detection [More on Description and Dataset](https://www.kaggle.com/c/deepfake-detection-challenge). The task at hand is pretty straightforward. Given a video, classify it into Real of Fake. But the Problem was the Dataset Size. It was roughly 300 GB. This was our source of reference on how to approach the problem [Link:](https://ieeexplore.ieee.org/document/8639163)

To Roughly Summarize, A pre-trained Inception V3 (CNN) is used to extract frame level features and are then fed to a LSTM.

![Data Pipeline](DataPipeline.png)

So, I planned to convert each of the videos (FRMS*X*Y*3) into (FRMS*2048). I've fed these videos to Pre-trained Inception V3 through a single step of forward propagation. Then, I've saved them as tfrecord format and these tfrecord files are used for training phase by the LSTM Network.

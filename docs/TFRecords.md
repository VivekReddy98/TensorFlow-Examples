<a id="top"></a>
# TFRecords Tutorial

**Contents**<br>
[What are TFRecords](#what-are-tfrecords)<br>
[Use Case: DeepFake Detection](#deepfake-detection) <br>
[Creating TFRecords](#where-to-put-it)<br>
[Feeding TFRecords for Model Training using tf.data](#writing-tests)<br>


## What are TFRecords

TF Records are Tensorflow's recommended and own binary storage format. When the training datasets are large, using TF Records would lead to a much better performance in feeding data to the model and thus leading to shorter training times. More about this on [Tensorflow.org](https://www.tensorflow.org/tutorials/load_data/tfrecord)

References to Well-written articles on TFRecords:
1. http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
2. https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564

## DeepFake-Detection

The code snippets in the next sections are taken from my other project related to DeepFake Detection [More on Description and Dataset](https://www.kaggle.com/c/deepfake-detection-challenge). [Repo](https://github.com/VivekReddy98/DeepFake-Detection) The task at hand is pretty straightforward. Given a video, classify it into Real of Fake. But the Problem was the Dataset Size. It was roughly 300 GB. 

This was the source of reference on how i choose to approach the problem [Link:](https://ieeexplore.ieee.org/document/8639163). To Roughly Summarize, A pre-trained Inception V3 (CNN) is used to extract frame level features and are then fed to a LSTM.

![Data Pipeline](DataPipeline.png)

So, I planned to convert each of the videos (FRMS*X*Y*3) into (FRMS*2048). I've fed these videos to Pre-trained Inception V3 through a single step of forward propagation. Then, I've saved them as tfrecord format and these tfrecord files are used for training phase by the LSTM Network.

## Creating TFRecords

Using TFRecords Essentially consists of two steps.
1) Converting any Input data format to a single .tfrecord format. This is not trivial becuase serialization has to be done manually.
2) Then Use TF Data API to decode and apply any data pre-processing techniques.

You can look at [src.video2tfrecordCustom.TFRecordGenerator](https://github.com/VivekReddy98/DeepFake-Detection/blob/70cc4edc5a234fd4823ef205d67cd7084bcad1a3/src/video2tfrecordCustom.py#L110) class for a comprehensive overview of the functionality.

For brevity, I'll just be focussing on TFRecord Conversion Method here.

```python
def save_video_as_tf_records_ylabels(self, file, label, split):

        # Pre-checks before reading a video file
        tfrecords_filename = os.path.join(self.OUT_PATH, file.split('.')[0] + "_" + label + "_" + split + '.tfrecords')
        check_filename = os.path.join(self.OUT_PATH, file.split('.')[0])
        file_exists = glob.glob(check_filename+"_*.tfrecords")
        if file_exists:
            print("{0} file aready exists".format(tfrecords_filename))
            return 0
        file_path = os.path.join(self.SRC_PATH, file)

        try:
            cap = cv2.VideoCapture(file_path) 
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if label == "FAKE":
                N_FRMS_PER_SAMPLE = self.N_FRMS_PER_SAMPLE_FAKE
            else:
                N_FRMS_PER_SAMPLE = self.N_FRMS_PER_SAMPLE_REAL

            # Allocate a Buffer to store extracted data from the Video File
            buf = np.empty((N_FRMS_PER_SAMPLE, self.WIDTH, self.HEIGHT, 3), np.dtype('float32'))

            if (frameCount < N_FRMS_PER_SAMPLE):
                print("Framecount too less to extract for file: " + tfrecords_filename + " frameCount: " + frameCount)
                return 0
            
            # Load the Video File into a Buffer
            fc = 0
            while (fc < N_FRMS_PER_SAMPLE):
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.central_crop(frame, 0.875)
                frame = cv2.resize(frame, (299, 299), interpolation = cv2.INTER_AREA)
                buf[fc] = tf.keras.applications.inception_v3.preprocess_input(frame)
                fc += 1
            buf.astype("float32")

            # Forward Pass through Inception V3 Pre-Trained Network, Output Shape: NUM_FRMS * 2048
            predictions = self.CNN_VECTORIZER.predict(buf)
            predictions = predictions.astype('float32') 

            
            # Converting into TF Records  
            options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP) # Gzip Compression
            writer = tf.python_io.TFRecordWriter(tfrecords_filename, options=options)

            if label == "FAKE":
                y_label = np.array([1, 0], dtype=np.int64)
            else:
                y_label = np.array([0, 1], dtype=np.int64)


            img_raw = predictions.tostring() # Convert to Bytes
            labels_raw = y_label.tostring() # Convert to Bytes

            # By Converting a 2D array into a Stream of Bytes, You'll loose the information of Shape and Datatype.
            # So, it's a good practice to save Shape Info. DataType info is fixed for "float32" at the other end. 
            # If a plain image jpg is used without any data preprocessing, you might want to to "int8", which could greatly reduce the memory footprint.
            example = tf.train.Example(features=tf.train.Features(feature={ 'vector_size': int64_feature(predictions.shape[1]),
                                                                            'batch_size': int64_feature(predictions.shape[0]),
                                                                            'image_raw': bytes_feature(img_raw),
                                                                            'labels_raw': bytes_feature(labels_raw)}))
            writer.write(example.SerializeToString()) # Write the Record
            writer.close()

        except Exception as e:
            print(str(e) + "for file: " + tfrecords_filename)

        return 1

```
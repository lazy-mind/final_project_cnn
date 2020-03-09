"""
Model definition for CNN sentiment training


"""

import os
import tensorflow as tf
import numpy as np
import sagemaker


def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling

    """

    embedding_matrix = np.zeros((config["embeddings_dictionary_size"], config["embeddings_vector_size"]))
    index = 0
    if config["cloud"] == 0:
        file = open(config["embeddings_path"], "r")
        for word_vector in file:
            if len(word_vector.split()) == config["embeddings_vector_size"]:
                # print(len(word_vector.split()))
                embedding_matrix[index, :] = word_vector.split()[0:]
            else:
                embedding_matrix[index, :] = word_vector.split()[1:]
            index += 1
    else:
        sess = sagemaker.Session()
        all_files = sess.list_s3_files(config["bucket"], config["embeddings_path"])
        result = sess.read_s3_file(config["bucket"], all_files[0])
        file = result.split("\n")[:-1]
        for word_vector in file:
            if len(word_vector.split()) == config["embeddings_vector_size"]:
                # print(len(word_vector.split()))
                embedding_matrix[index, :] = word_vector.split()[0:]
            else:
                embedding_matrix[index, :] = word_vector.split()[1:]
            index += 1

    # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    model = tf.keras.models.Sequential()

    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
    model.add(tf.keras.layers.Embedding(input_length=config["padding_size"],
                        input_dim=config["embeddings_dictionary_size"],
                        output_dim=config['embeddings_vector_size'],
                        weights=[embedding_matrix],
                        # embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                        name=config["embedding_layer_name"],
                        trainable=True))


    model.add(tf.keras.layers.SpatialDropout1D(0.6))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1))








    # 1-layer LSTM
    # model.add(tf.keras.layers.LSTM(32))
    # model.add(tf.keras.layers.Dense(24, activation='relu'))
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))




    # 2-layer lSTM
    # model.add(tf.keras.layers.SpatialDropout1D(0.5))
    # model.add(tf.keras.layers.LSTM(32, return_sequences=True))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.LSTM(16))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(8, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))




    # # GRU
    # model.add(tf.keras.layers.GRU(32))
    # model.add(tf.keras.layers.Dense(24, activation='relu'))




    # Standard CNN
    # model.add(tf.keras.layers.Conv1D(64, kernel_size=2, strides=1, activation='relu'))
    # model.add(tf.keras.layers.GlobalMaxPool1D())
    # model.add(tf.keras.layers.Dense(100, activation='relu'))
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



    # Standard Dropout
    # model.add(tf.keras.layers.Conv1D(64, kernel_size=2, strides=1, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.GlobalMaxPool1D())
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(100, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



    # # Standard Multi Conv Layer
    # model.add(tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.GlobalMaxPool1D())
    # model.add(tf.keras.layers.Dropout(0.5))
    # # model.add(tf.keras.layers.Conv1D(64, kernel_size=2, strides=1, activation='relu'))
    # # model.add(tf.keras.layers.Dropout(0.5))
    # # model.add(tf.keras.layers.GlobalMaxPool1D())
    # # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(100, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))













    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    cnn_model = model

    print(model.summary())

    return cnn_model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving


    """
    print("saving model")

    tf.saved_model.save(model, os.path.join(output, "1"))

    print("Model successfully saved at: {}".format(output))

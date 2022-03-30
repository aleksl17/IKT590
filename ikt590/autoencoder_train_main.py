import numpy as np
import tensorflow
import datetime
import random
import os

import helpers.autoencoder_models as autoencoder_models
import helpers.data_manipulation as data_manipulation


# Disables GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    # Variables
    batch_size = 10000
    batch_epoch = 100
    batches = 10
    epochs = 250
    sample_size = 40
    output_size = 3
    load_old_models = False
    batch = False
    loss_list = []
    dataset = []
    x = []

    meta, dataset = data_manipulation.read_dataset(datasetFile='datasets/V2.0/dataset.json')

    dataset = np.asarray(dataset)

    for d in dataset:
        minVal = min(d)*0.9
        x.append((d - minVal)/(max(d)-minVal))

    # Define models
    if load_old_models:
        encoder = tensorflow.keras.models.load_model('models/encoder')
        decoder = tensorflow.keras.models.load_model('models/decoder')
    else:    
        encoder, decoder = autoencoder_models.getModels(sample_size, output_size)
        encoder._name = "encoder"
        decoder._name = "decoder"

    model = tensorflow.keras.models.Sequential()
    model.add(encoder)
    model.add(decoder)
    model.compile(optimizer='adam', loss='mse')

    if batch:
        # Batches
        for i in range(batches):
            print('_________________________________________________________________')
            print(f'Training for batch: {i} of {batches}')
            print(f'Total training epochs: {i * batch_epoch}. Total training data: {i*batch_size}')
            print(f'Time: {datetime.datetime.now()}')

            x0 = np.array(random.sample(x,batch_size))
            history = model.fit(x0.reshape(x0.shape[0], x0.shape[1], 1),x0, epochs=batch_epoch, shuffle=True, verbose=1)
            loss = sum(history.history['loss']) / len(history.history['loss'])

            print(f'Loss for batch {i}: {loss }')
            print(f'Saving models for batch: {i}')

            encoder.save('models/encoder')
            decoder.save('models/decoder')
    else:
        # All data
        x = np.asarray(x)
        model.fit(x.reshape(x.shape[0], x.shape[1], 1), x, epochs=epochs, shuffle=True, verbose=1)
        encoder.save('models/encoder')
        decoder.save('models/decoder')


if __name__ == "__main__":
    main()

import numpy as np
import datetime
import random
import os

from tensorflow.keras.models import Sequential, load_model

import helpers.autoencoder_models as autoencoder_models
import helpers.data_manipulation as data_manipulation


# Disables GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    # Variables
    batches = 10
    batch_size = 10000
    batch_epoch = 1000
    sample_size = 40
    output_size = 3
    load_old_models = False
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
        encoder = load_model('models/encoder')
        decoder = load_model('models/decoder')
    else:    
        encoder, decoder = autoencoder_models.getModels(sample_size, output_size)
        encoder._name = "encoder"
        decoder._name = "decoder"

    model = Sequential()
    model.add(encoder)
    model.add(decoder)
    model.compile(optimizer='adam', loss='mse')

    for i in range(batches):
        print('_________________________________________________________________')
        print(f'Training for batch: {i} of {batches}')
        print(f'Total training epochs: {i * batch_epoch}. Total training data: {i*batch_size}')
        print(f'Time: {datetime.datetime.now()}')
        x0 = np.array(random.sample(x,batch_size))
        history = model.fit(x0.reshape(x0.shape[0], x0.shape[1], 1),x0, epochs=batch_epoch, shuffle=True, verbose=0)
        loss = sum(history.history['loss']) / len(history.history['loss'])

        print(f'Loss for batch {i}: {loss }')
        loss_list.append(loss)

        print(f'Saving models for batch: {i}')
        encoder.save('models/encoder')
        decoder.save('models/decoder')


if __name__ == "__main__":
    main()

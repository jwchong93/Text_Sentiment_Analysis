import numpy as np
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow as tf

dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

BATCH_SIZE = 64
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

model = keras.models.load_model("IMDB_review_model")
sample_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = model.predict(np.array([sample_text]))
print(predictions)

input_text = input('Enter your comment:')
while input_text != '!stop':
    predictions = model.predict(np.array([input_text]))
    print(predictions)
    if predictions > 0:
        print("This is a positive comment")
    else:
        print("This is a negative comment")
    input_text = input('Enter your comment:')


import numpy as np
import tensorflow as tf
import tensorflow.keras.saving 

model = tensorflow.keras.saving.load_model("/Users/leopoldotodisco/TomatoCNN/models/small_CNN.keras")
class_names = ["Damaged", "Old", "Ripe", "Unripe"]

def get_prediction(img):
    
    logits = model(img, training=False)
    probs = tf.nn.softmax(logits, axis=1)  # softmax esplicita

    pred_idx = tf.argmax(probs, axis=1).numpy()[0]  # converto in int
    pred_prob = probs[0, pred_idx].numpy()

    print("Probabilità:", probs.numpy())
    print("pred prob:", pred_prob)
    print("pred prob:", class_names[pred_idx])

    return class_names[pred_idx]

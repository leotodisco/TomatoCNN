import numpy as np
import tensorflow as tf
import tensorflow.keras.saving 
from dotenv import find_dotenv, load_dotenv
import os

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

model = tensorflow.keras.saving.load_model(os.getenv("model_path"))
class_names = ["Damaged", "Old", "Ripe", "Unripe"]

def get_prediction(img):
    
    logits = model(img, training=False)
    probs = tf.nn.softmax(logits, axis=1)

    pred_idx = tf.argmax(probs, axis=1).numpy()[0]
    pred_prob = probs[0, pred_idx].numpy()

    print("All probs:", probs.numpy())
    print("pred prob:", pred_prob)
    print("pred class:", class_names[pred_idx])

    return class_names[pred_idx]

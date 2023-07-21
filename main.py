from tensorflow.keras.models import load_model
from training import build_and_train_models
import tensorflow as tf

# By saving the weights, it allows you to improve and retrain the model
"""
the original model used 2000 steps, here I just load a small number of steps
to highlight the process, and get the model archetecture to load the weights into..
..if you want to retrain the model add more steps than 10 and don't load the weights.
"""
steps = 10  # original model used 2000 (choose a low number if you)
gen0, gen1 = build_and_train_models(train_steps=steps)

# Comment and uncomment to load/save models
# gen1.save("gen0")
# gen0.save_weights("gen1_weights")

# Load the fully developed model.

gen0 = load_model("gen0")
gen1.load_weights('gen1_weights')

tf.keras.utils.plot_model(gen1, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB',
                          expand_nested=False, dpi=96)

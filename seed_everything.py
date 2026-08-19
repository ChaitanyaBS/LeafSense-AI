# seed_everything.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses TF info/warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppresses oneDNN info log
"""
LeafSense AI - Reproducibility & Seeding Module
Ensures reproducible results across Python, NumPy, and TensorFlow.
"""

import os
import random
import numpy as np
import tensorflow as tf

def seed_everything(seed=42):
    """
    Sets seeds for Python random, NumPy, and TensorFlow to ensure reproducible results.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    print(f"[INFO] Random seed set to {seed} for reproducibility.")

if __name__ == "__main__":
    seed_everything(42)

# # Seed Everything to reproduce results for future use cases
# def seed_everything(seed=42):
#     # Seed value for TensorFlow
#     tf.random.set_seed(seed)
#
#     # Seed value for NumPy
#     np.random.seed(seed)
#
#     # Seed value for Python's random library
#     random.seed(seed)
#
#     # Force TensorFlow to use single thread
#     # Multiple threads are a potential source of non-reproducible results.
#     session_conf = tf.compat.v1.ConfigProto(
#         intra_op_parallelism_threads=1,
#         inter_op_parallelism_threads=1
#     )
#
#     # Make sure that TensorFlow uses a deterministic operation wherever possible
#     tf.compat.v1.set_random_seed(seed)
#
#     sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
#     tf.compat.v1.keras.backend.set_session(sess)
#
#
# seed_everything()
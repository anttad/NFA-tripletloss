import io
import os 
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
import argparse

# import tensorflow_datasets as tfds
from sklearn.utils import shuffle

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.keras.backend.set_floatx('float64')


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int,
        help='number of epoch', default=1000)
parser.add_argument('--batch_size', type=int,
        help='size of the batch.', default=32)
parser.add_argument('--margin', type=float,
        help='margin for the triplet loss', default=0.3)
parser.add_argument('--n_dim', type=int,
        help='embedding dimension', default=128)
parser.add_argument('--model_fold', type=str,
        help='folder where to save the model', default="/code/models/")
parser.add_argument('--emb_fold', type=str,
        help='folder where to save the model', default="/code/embeddings/")
parser.add_argument('--log_fold', type=str,
        help='folder where to save the model', default="/code/tensorboard/")

parser.add_argument('--dist_fn', type=str,
        help='Distance function, choose between "cosine", "angular" and "L2"', default="L2")

args = parser.parse_args()

margin = args.margin
n_dim = args.n_dim
epoch = args.epoch
batch_size = args.batch_size

log_dir = os.path.join(args.log_fold,"dist_{}_margin_{}_{}D".format(args.dist_fn,args.margin,n_dim))


def spherical_cap_hypothesis(feature):
    #r=1
    feature = tf.math.l2_normalize(feature, axis=1)
    tf.cast(feature,tf.float64)
    # n =  feature.shape[1] 
    n = tf.constant(feature.shape[1], dtype=tf.float64)
    print("================================= n: ",feature.shape[1] )
    eps = tf.keras.backend.epsilon()
    theta = tf.acos(tf.clip_by_value(tf.matmul(feature, feature, transpose_b=True),-1+eps,1-eps))
    print(theta)
    
    #print("angle min max: ", theta.min(),theta.max(), theta.shape)
    #h_tilde = x1_norm@x2_norm.T
  
    # n = n_dim
    
    if tf.math.is_nan(feature):
        raise(ValueError("NaN in feature computation"))
    if tf.math.is_nan(theta):
        raise(ValueError("NaN in theta computation"))
    A=0.5* tf.math.betainc(0.5*(n-1),0.5,tf.sin(theta)**2)
    if tf.math.is_nan(A):
        raise(ValueError("NaN in area computation"))
    tf.cast(A,tf.float32)
    return A

def spherical_cap_hypothesis_nfa(feature,n_dim=args.n_dim,N_test=1000):
    #r=1
    feature = tf.math.l2_normalize(feature, axis=1)
    eps = tf.keras.backend.epsilon()
    theta = tf.acos(tf.clip_by_value(tf.matmul(feature, feature, transpose_b=True),-1+eps,1-eps))

    
    #print("angle min max: ", theta.min(),theta.max(), theta.shape)
    #h_tilde = x1_norm@x2_norm.T
  
    n = n_dim
    A=0.5* tf.math.betainc(0.5*(n-1),0.5,tf.sin(theta)**2)
    NFA = N_test*A
    to_log10 = tf.math.log(10)

    logNFA =  tf.math.log(NFA)/to_log10

    return logNFA


@tf.function
def angular_distance(feature):
    """Computes the angular distance matrix.
    output[i, j] = 1 - cosine_similarity(feature[i, :], feature[j, :])
    Args:
      feature: 2-D Tensor of size `[number of data, feature dimension]`.
    Returns:
      angular_distances: 2-D Tensor of size `[number of data, number of data]`.
    """
    # normalize input
    feature = tf.math.l2_normalize(feature, axis=1)
    eps = tf.keras.backend.epsilon()
    # create adjaceny matrix of cosine similarity
    # angular_distances = tf.math.acos(tf.matmul(feature, feature, transpose_b=True))
    # print(tf.reduce_mean(tf.math.is_nan(angular_distances)))
    # input()
    angular_distances = tf.acos(tf.clip_by_value(tf.matmul(feature, feature, transpose_b=True),-1+eps,1-eps))

    # ensure all distances > 1e-16
    # angular_distances = tf.maximum(angular_distances, 0.0)
    return angular_distances



@tf.function
def cosine_distance(feature):
    """Computes the angular distance matrix.
    output[i, j] = 1 - cosine_similarity(feature[i, :], feature[j, :])
    Args:
      feature: 2-D Tensor of size `[number of data, feature dimension]`.
    Returns:
      angular_distances: 2-D Tensor of size `[number of data, number of data]`.
    """
    # normalize input
    feature = tf.math.l2_normalize(feature, axis=1)

    # create adjaceny matrix of cosine similarity
    angular_distances = tf.matmul(feature, feature, transpose_b=True)
    # print(tf.reduce_mean(tf.math.is_nan(angular_distances)))
    # input()
    # angular_distances = tf.math.acos(tf.clip_by_value(tf.matmul(feature, feature, transpose_b=True),-1,1))

    # ensure all distances > 1e-16
    angular_distances = tf.maximum(angular_distances, 0.0)
    return angular_distances


# test = np.array([[1,2,3],[1,2,4]])

# print(test)
# print(cosine_distance(tf.convert_to_tensor((test))))
# print(test)
# input()


def cos_dist():
    pass

def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)

def normalize(img):
    return img/255.
train_dataset, test_dataset = tf.keras.datasets.mnist.load_data()

X_train, y_train = shuffle(*train_dataset)
print(X_train.shape)
X_test, y_test = test_dataset
X_train = np.expand_dims(normalize(X_train),axis=-1)
X_test = np.expand_dims(normalize(X_test), axis=-1)


#train_dataset, test_dataset = tfds.load(name="mnist", split=['train', 'test'], as_supervised=True)

# Build your input pipelines
# train_dataset = train_dataset.shuffle(1024).batch(32)
# train_dataset = train_dataset.map(_normalize_img)

# test_dataset = test_dataset.batch(32)
# test_dataset = test_dataset.map(_normalize_img)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)),
    # tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
    # tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=8, kernel_size=2, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=4, kernel_size=2, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(n_dim, activation=None), # No activation on final dense layer
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings

])

# Compile the model
print("compiling")
if args.dist_fn == "cosine":
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss(distance_metric="angular",margin=margin))

elif args.dist_fn == "L2":
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss(distance_metric="L2",margin=margin))

elif args.dist_fn == "angular":
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss(distance_metric=angular_distance,margin=margin))
elif args.dist_fn == "sphere":
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss(distance_metric=spherical_cap_hypothesis,margin=margin))
  

print(model.summary())
# Train the network
# history = model.fit(
#     train_dataset,
#     epochs=5)
print("training")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=50,write_images=True,embeddings_freq=100)
history = model.fit(
    X_train,y_train,
    epochs=epoch, batch_size=batch_size, callbacks=[tensorboard_callback])

# Evaluate the network
results = model.predict(X_test)
np.save(os.path.join(args.emb_fold,"test_dist_{}_margin_{}_{}D.npy".format(args.dist_fn,args.margin,n_dim)),results)
results = model.predict(X_train)
np.save(os.path.join(args.emb_fold,"train_dist_{}_margin_{}_{}D.npy".format(args.dist_fn,args.margin,n_dim)),results)
# Save test embeddings for visualization in projector
np.savetxt("vecs.tsv", results, delimiter='\t')

# out_m = io.open('meta.tsv', 'w', encoding='utf-8')
# for img, labels in tfds.as_numpy(test_dataset):
#     [out_m.write(str(x) + "\n") for x in labels]
# out_m.close()
tf.keras.models.save_model(model,os.path.join(args.model_fold,"dist_{}_margin_{}_{}D.h5".format(args.dist_fn,args.margin,n_dim)))

try:
  from google.colab import files
  files.download('vecs.tsv')
  files.download('meta.tsv')
except:
  pass

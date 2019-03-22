import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.set_session(sess)

# import packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required = True,
    help = "path to HDF5 dataset")
ap.add_argument("-m", "--model", required = True,
    help = "path to output model")
ap.add_argument("-j", "--jobs", type = int, default = -1,
    help = "# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

# open HDF5 database for reading
# then determine the index of the training and testing split
db = h5py.File(args["db"], mode = "r")
i = int(db["labels"].shape[0] * 0.75)

# define the set of parameters to tune when we start a grid search
print("[INFO] tuning hyperparameters...")
params = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}
model = GridSearchCV(LogisticRegression(), params, cv = 3, n_jobs = args["jobs"])
model.fit(db["features"][:i], db["labels"][:i])
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# evaluate the model
print("[INFO] evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds,
    target_names = db["label_names"]))

# serialize the model to disk
print("[INFO] saving model...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

# close the database
db.close()

# This file trains and saves the trained model to the disk.
# Loss curves and metrics are saved.

# Author M. Usman Rafique

## Imports
import os
import keras
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from dataset import MappingChallengeDataset
from keras.losses import binary_crossentropy

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from model import get_model
from losses_metrics import disagg_loss, empty_loss, my_IoU, my_acc
from dataset import data_generator_modified
from config import Config as CFG


## Prepare datasets
# Load training dataset
dataset_train = MappingChallengeDataset()
dataset_train.load_dataset(dataset_dir=os.path.join(CFG.ROOT_DIR, "train"), load_small=CFG.SMALL_DATA)
dataset_train.prepare()

# Load validation dataset
dataset_val = MappingChallengeDataset()
val_coco = dataset_val.load_dataset(dataset_dir=os.path.join(CFG.ROOT_DIR, "val"), load_small=CFG.SMALL_DATA, return_coco=True)
dataset_val.prepare()

# Prepare generators
train_generator = data_generator_modified(dataset_train, CFG, shuffle=True,
                                         augmentation=None,
                                         batch_size=CFG.BATCH_SIZE)
val_generator = data_generator_modified(dataset_val, CFG, shuffle=True, batch_size=CFG.BATCH_SIZE)

print('Generator objects have been created')

## Get the U-Net model
model = get_model() # get a U-net model

# Optimizer
my_opt = optimizers.Adam(lr=0.0005, decay=0.0)

# Get loss function
if CFG.LOSS_FN == 'CE':
    main_loss = binary_crossentropy
elif CFG.LOSS_FN == 'Proposed_OneSided':
    main_loss = disagg_loss

keras.losses.main_loss = main_loss

# Compile model
model.compile(loss={'output_original': 'main_loss', 'placeholder_conv':'empty_loss', 'aux_id':'empty_loss'}, optimizer= my_opt,
              metrics={'output_original': ['my_IoU','my_acc', 'accuracy']})

print('Model prepared...')
model.summary()

directory = CFG.DIRECTORY

if not os.path.exists(directory):
    os.makedirs(directory)
else:
    print('Folder already exists. Are you sure you want to overwrite results?')
    print('Debug')

# Callbacks
myCheckPoint = ModelCheckpoint(filepath = 'trained_model.hdf5', save_best_only=True, monitor='output_original_my_IoU', mode='max')
earlyStopping = EarlyStopping(patience=10)
logfile_name = os.path.join(directory,'training_log.csv')
csvLogger = CSVLogger(logfile_name, separator=',', append=False)

my_step = int(np.floor(len(dataset_train.image_ids) / CFG.BATCH_SIZE))  # Training step size

## Training
history = model.fit_generator(train_generator,
                             epochs=CFG.num_epoch,
                             steps_per_epoch= my_step, #Config.STEPS_PER_EPOCH,
                             validation_data= val_generator, #class_weight= {'uNet_output': ['my_class_weight']} ,
                             validation_steps= CFG.VALIDATION_STEPS, #workers=myworkers,
                             callbacks= [myCheckPoint, earlyStopping, csvLogger],  # reduce_lr
                             verbose=True)

# Load the best checkpoint
model.load_weights('trained_model.hdf5')

model.save(os.path.join(directory, 'trained_model_full.h5'))

# implementing length
my_step = int(np.floor(len(dataset_val.image_ids) / CFG.BATCH_SIZE))

score = model.evaluate_generator(val_generator, steps = my_step)
print('Model evaluation results: '+ str(score))
print(model.metrics_names)

print('Here is the training history:')
print(history.history)
f = open(os.path.join(directory,'history.txt'), 'w')
f.write(str(history.history))
f.close()

plt.figure()
plt.plot(history.history['output_original_loss'])
plt.plot(history.history['val_output_original_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(directory,'loss.png'))
plt.show()


plt.figure()
plt.plot(history.history['uNet_output_my_IoU'])
plt.plot(history.history['val_uNet_output_my_IoU'])
plt.title('IoU')
plt.ylabel('IoU')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(directory,'IoU.png'))
plt.show()

plt.figure()
plt.plot(history.history['uNet_output_my_acc'])
plt.plot(history.history['val_uNet_output_my_acc'])
plt.title('Pixel Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(directory,'Building_Accuracy.png'))
plt.show()

plt.figure()
plt.plot(history.history['uNet_output_acc'])
plt.plot(history.history['val_uNet_output_acc'])
plt.title('Default Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(directory,'default_accuracy.png'))
plt.show()

print('Finished training. Trained model, loss curves, and metrics have been stored on the disk')
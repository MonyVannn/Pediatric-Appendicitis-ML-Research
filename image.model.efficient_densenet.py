# image.model.efficient_densenet.py
# This script mirrors the pipeline in image.model.resnet.ipynb but trains EfficientNetB7, DenseNet121 and DenseNet201.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K

# App-specific imports for architectures
from tensorflow.keras.applications import DenseNet121, DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input as eff_preprocess_input

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Paths - adjust if needed
appendicitis_path = "images/appendicitis_images/"
no_appendicitis_path = "images/no_appendicitis_images/"

# Data loading & preprocessing (same as notebook)
def load_and_preprocess_data(img_size=(224, 224)):
    images = []
    labels = []

    appendicitis_files = [f for f in os.listdir(appendicitis_path) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
    print(f"Loading {len(appendicitis_files)} appendicitis images...")
    for filename in appendicitis_files:
        try:
            img = Image.open(os.path.join(appendicitis_path, filename))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(1)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    no_appendicitis_files = [f for f in os.listdir(no_appendicitis_path) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
    print(f"Loading {len(no_appendicitis_files)} no-appendicitis images...")
    for filename in no_appendicitis_files:
        try:
            img = Image.open(os.path.join(no_appendicitis_path, filename))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(0)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    X = np.array(images)
    y = np.array(labels)
    print(f"Loaded {len(X)} images with shape {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    return X, y

# Build backbone mapping
def build_backbone_model(arch_name, input_shape=(224,224,3), base_trainable=False):
    arch_name = arch_name.lower()
    models = {
        'efficientnetb7': (EfficientNetB7, eff_preprocess_input),
        'densenet121': (DenseNet121, densenet_preprocess_input),
        'densenet201': (DenseNet201, densenet_preprocess_input)
    }
    if arch_name not in models:
        raise ValueError(f"Unknown architecture: {arch_name}")

    ModelClass, preprocess_fn = models[arch_name]
    base = ModelClass(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = base_trainable
    return base, preprocess_fn

# Build full model with custom head (same head as notebook)
def build_full_model(arch_name, input_shape=(224,224,3)):
    base_model, preprocess_fn = build_backbone_model(arch_name, input_shape=input_shape, base_trainable=False)
    x_input = keras.Input(shape=input_shape, name='input_image')
    if preprocess_fn is not None:
        x = keras.layers.Lambda(lambda t: t * 255.0)(x_input)
        x = preprocess_fn(x)
    else:
        x = x_input
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=x_input, outputs=output)
    return model, base_model

# Metrics, compile and utility functions (copied from notebook)
def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        pp = K.sum(K.round(K.clip(y_true, 0, 1)))
        return tp / (pp + K.epsilon())
    def precision(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return tp / (pred_pos + K.epsilon())
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def compile_model_phase(model, learning_rate=1e-3, use_f1=True):
    metrics_list = ['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
    if use_f1:
        metrics_list.append(f1_score)
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=metrics_list)

def get_callbacks(arch_name, phase, monitor_metric='val_accuracy'):
    fname = f'best_{arch_name}_phase{phase}.keras'
    return [
        EarlyStopping(monitor='val_loss', patience=8 if phase==2 else 10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3 if phase==2 else 5, min_lr=1e-8, verbose=1),
        ModelCheckpoint(fname, monitor=monitor_metric, mode='max', save_best_only=True, verbose=1)
    ]

def unfreeze_for_finetune(base_model, arch_name):
    arch = arch_name.lower()
    # sensible defaults; these can be tuned
    if 'efficientnet' in arch:
        layers_to_unfreeze = 80  # unfreeze last ~80 layers (EfficientNet is deep)
    elif 'densenet121' in arch:
        layers_to_unfreeze = 40
    elif 'densenet201' in arch:
        layers_to_unfreeze = 60
    else:
        layers_to_unfreeze = 20
    for layer in base_model.layers[:-layers_to_unfreeze]:
        layer.trainable = False
    for layer in base_model.layers[-layers_to_unfreeze:]:
        layer.trainable = True

def evaluate_and_report(model, X_data, y_data, set_name='Validation'):
    preds = model.predict(X_data, verbose=0)
    binary_preds = (preds > 0.5).astype(int).ravel()
    report = classification_report(y_data, binary_preds, target_names=['No Appendicitis','Appendicitis'])
    print(f"\n--- {set_name} Classification Report ---")
    print(report)
    cm = confusion_matrix(y_data, binary_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + K.epsilon())
    specificity = tn / (tn + fp + K.epsilon())
    print(f"Sensitivity (Recall): {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    return { 'confusion_matrix': cm, 'sensitivity': sensitivity, 'specificity': specificity, 'classification_report': report }

# Main execution: load data, split, compute class weights
if __name__ == '__main__':
    X, y = load_and_preprocess_data(img_size=(224,224))
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1, fill_mode='nearest')
    val_test_datagen = ImageDataGenerator()

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print('Class weights:', class_weight_dict)

    architectures = ['efficientnetb7', 'densenet121', 'densenet201']
    for arch in architectures:
        print('\n' + '='*50)
        print(f' TRAINING ARCHITECTURE: {arch.upper()}')
        print('='*50)
        model, base_model = build_full_model(arch, input_shape=(224,224,3))
        compile_model_phase(model, learning_rate=1e-3, use_f1=True)
        # Phase 1: train top layers only
        callbacks = get_callbacks(arch, phase=1, monitor_metric='val_accuracy')
        model.fit(train_datagen.flow(X_train, y_train, batch_size=16), epochs=20, validation_data=(X_val, y_val), class_weight=class_weight_dict, callbacks=callbacks, verbose=1)
        # Phase 2: unfreeze some backbone and fine-tune
        unfreeze_for_finetune(base_model, arch)
        compile_model_phase(model, learning_rate=1e-4, use_f1=True)
        callbacks = get_callbacks(arch, phase=2, monitor_metric='val_accuracy')
        model.fit(train_datagen.flow(X_train, y_train, batch_size=16), epochs=15, validation_data=(X_val, y_val), class_weight=class_weight_dict, callbacks=callbacks, verbose=1)
        # Final evaluation
        print(f"\n--- Final Evaluation for {arch.upper()} ---")
        evaluate_and_report(model, X_val, y_val, set_name='Validation')
        evaluate_and_report(model, X_test, y_test, set_name='Test')

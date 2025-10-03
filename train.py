import os
import glob
import argparse
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Add,
    Dropout,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def residual_block(x, filters, stride=1, conv_shortcut=False):
    """A residual block for ResNet - from Agri_AI (2).ipynb"""
    shortcut = x
    x = Conv2D(filters, kernel_size=(3, 3), strides=stride, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    if conv_shortcut:
        shortcut = Conv2D(filters, kernel_size=(1, 1), strides=stride, padding="same", kernel_initializer="he_normal")(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def build_resnet_from_scratch(input_shape, num_classes):
    """Builds a simplified ResNet-like model from scratch - from Agri_AI (2).ipynb"""
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same", kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=128, stride=2, conv_shortcut=True)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=256, stride=2, conv_shortcut=True)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=512, stride=2, conv_shortcut=True)
    x = residual_block(x, filters=512)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu", kernel_initializer="he_normal")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model


def discover_classes(dataset_dir):
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    if "Background_without_leaves" in class_names:
        class_names.remove("Background_without_leaves")
    return class_names


def build_generators(dataset_dir, image_size=(128, 128), batch_size=32):
    class_names = discover_classes(dataset_dir)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
        classes=class_names,
    )

    val_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
        classes=class_names,
    )

    return train_generator, val_generator, class_names


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.keras"))
    if not checkpoints:
        return None, 0
    checkpoints.sort(key=os.path.getmtime)
    latest_checkpoint = checkpoints[-1]
    try:
        base = os.path.basename(latest_checkpoint)
        parts = base.split("_")
        epoch_str = parts[5]
        start_epoch = int(epoch_str)
    except Exception:
        start_epoch = 0
    return latest_checkpoint, start_epoch + 1


def save_classes(classes, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for cls in classes:
            f.write(cls + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Agri AI model")
    parser.add_argument("--data-dir", required=True, help="Path to dataset root directory")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--logs-dir", default="runs", help="Directory to store TensorBoard logs")
    parser.add_argument("--epochs", type=int, default=150, help="Total epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128], help="Image size H W")
    parser.add_argument("--final-model-path", default="models/final_model.keras", help="Path to save final model")
    parser.add_argument("--classes-path", default="models/classes.txt", help="Path to save classes.txt")

    args = parser.parse_args()

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.final_model_path), exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    image_size = tuple(args.image_size)

    train_gen, val_gen, class_names = build_generators(
        dataset_dir=args.data_dir, image_size=image_size, batch_size=args.batch_size
    )

    save_classes(class_names, args.classes_path)

    num_classes = len(class_names)

    latest_ckpt, initial_epoch = get_latest_checkpoint(args.checkpoints_dir)
    if latest_ckpt:
        print(f"Loading model from checkpoint: {latest_ckpt}")
        model = load_model(latest_ckpt, compile=False)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    else:
        print("No existing checkpoints found. Starting training from scratch.")
        model = build_resnet_from_scratch(input_shape=image_size + (3,), num_classes=num_classes)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
        initial_epoch = 0

    checkpoint_filepath = os.path.join(
        args.checkpoints_dir, "agri_ai_scratch_v2_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras"
    )
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    log_dir = os.path.join(args.logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
    early_stopping_callback = EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1)
    reduce_lr_callback = ReduceLROnPlateau(monitor="val_accuracy", factor=0.2, patience=7, min_lr=0.000001, verbose=1)

    callbacks = [model_checkpoint_callback, early_stopping_callback, reduce_lr_callback, tensorboard_callback]

    print("\n" + "=" * 50)
    print("  Agri AI Training Started  ")
    print("=" * 50)

    model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        steps_per_epoch=train_gen.samples // args.batch_size,
        validation_steps=val_gen.samples // args.batch_size,
    )

    print("\nTraining Finished!")
    model.save(args.final_model_path)
    print(f"Final model saved to: {args.final_model_path}")


if __name__ == "__main__":
    main()



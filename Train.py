import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Nadam

# Configuration
IMAGE_SIZE = 256  # Image dimensions (height, width)
BATCH_SIZE = 10  # Number of images processed in each batch
EPOCHS = 20  # Number of training epochs

# Paths - Adjust these for your specific case
TRAIN_DIR = 'classes'  # Directory containing training data, organized by class
VALIDATION_DIR = 'TestClass'  # Directory containing validation data, organized by class
MODEL_PATH = 'resNet_model_for_fingerprintClassification.h5'  # Path to pre-trained or new model file. If not present, it will download and train ResNet50.
SAVE_PATH_PKL = 'ResNet_fingerprintClassification.pkl'  # Path to save the model in .pkl format
SAVE_PATH_H5 = 'ResNet_fingerprintClassification.h5'  # Path to save the model in .h5 format

# Check if model exists
if os.path.exists(MODEL_PATH):
    print("Model found. Loading the pre-trained model.")
    model = load_model(MODEL_PATH)
else:
    print("Model not found. Initializing and training a new model.")

    # Data Generators
    train_generator = ImageDataGenerator(
        featurewise_center=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input
    )

    validation_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_generator.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_gen = validation_generator.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Initialize ResNet50 Model with Transfer Learning
    base_model = ResNet50(include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(5, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the Model
    model.compile(
        optimizer=Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
        metrics=['accuracy'],
        loss='categorical_crossentropy'
    )

    # Train the Model
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=EPOCHS,
        validation_data=validation_gen,
        validation_steps=len(validation_gen)
    )

    # Save the Model
    model.save(MODEL_PATH)
    model.save(SAVE_PATH_H5)
    model.save(SAVE_PATH_PKL)
    print("Model trained and saved at:", MODEL_PATH)
    print("Model additionally saved as .h5 and .pkl at:", SAVE_PATH_H5, "and", SAVE_PATH_PKL)

# Evaluate the Model
score_training = model.evaluate(validation_gen, steps=len(validation_gen), verbose=0)
print("Loss of trained model:", score_training[0])
print("Accuracy of trained model:", score_training[1])

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='center right')
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Test the Model
from tensorflow.keras.preprocessing import image 

test_image_path = r'haris\right_ring.jpg'  # Path to the test image
test_image = image.load_img(test_image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
predictions = model.predict(test_image)

print("Predictions:")
print("Arch:", (predictions[0][0]) * 100, "%")
print("Left Loop:", (predictions[0][1]) * 100, "%")
print("Right Loop:", (predictions[0][2]) * 100, "%")
print("Tented Arch:", (predictions[0][3]) * 100, "%")
print("Whirl:", (predictions[0][4]) * 100, "%")

# Display Predictions
print("Class Indices:", train_gen.class_indices)

# Save the Model in Additional Formats
model.save(SAVE_PATH_PKL)
print("Model additionally saved as .pkl at:", SAVE_PATH_PKL)

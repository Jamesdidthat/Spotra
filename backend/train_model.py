import tensorflow as tf

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5  # Adjust the number of epochs for fine-tuning

# Load the existing model
model = tf.keras.models.load_model("movie_recognition_model.h5")

# Prepare dataset (before applying transformations)
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ✅ Get class names before mapping transformations
class_names = raw_train_ds.class_names
num_classes = len(class_names)

# Normalize the data
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = raw_train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = raw_val_ds.map(lambda x, y: (normalization_layer(x), y))

#Remove the last layer and replace it with a new one
model.layers.pop()  # Remove the last layer
model.add(tf.keras.layers.Dense(num_classes, activation='softmax', name="output_layer"))  # Add new final layer

# Recompile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Save the updated model
model.save("movie_recognition_model_updated.h5")

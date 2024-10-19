import cv2
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from imutils import paths
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet101, Xception
from tensorflow.keras.layers import Input, Dense, Flatten, AveragePooling2D, Dropout, BatchNormalization, Conv2D

# Set parameters
LR = 0.001
EPOCHS = 10
BATCH_SIZE = 32
INP_SIZE = (224, 224, 3)
NORMAL_LEN = 218  # The length for normal and pneumonia images
classes = ['normal', 'pneumonia']


# Function to load images and preprocess them
def create_data(dir_name, limit):
    temp_data = []
    img_list = os.listdir(dir_name)
    for img_name in img_list[:limit]:
        img_path = os.path.join(dir_name, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        temp_data.append(image)
    return temp_data


# Load data for 'normal' and 'pneumonia' classes
data = []
labels = []

normal_dir = 'COVID-19 Radiography Database/NORMAL/'
pneumonia_dir = 'COVID-19 Radiography Database/Viral Pneumonia/'

data.extend(create_data(normal_dir, NORMAL_LEN))
data.extend(create_data(pneumonia_dir, NORMAL_LEN))

# Label the classes: 0 for normal and 1 for pneumonia
labels.extend([0] * NORMAL_LEN)  # Normal
labels.extend([1] * NORMAL_LEN)  # Pneumonia

# Normalize data and convert labels to categorical
data = np.array(data) / 255.0
labels = np.array(labels)

# One-hot encode labels
labels = to_categorical(labels)

# Split data into training and test sets
(x_train, x_test, y_train, y_test) = train_test_split(
    data,
    labels,
    test_size=0.20,
    stratify=labels
)

# Data augmentation
trainAug = ImageDataGenerator(
    rotation_range=15,
    fill_mode="nearest"
)


# Function to generate a custom model
def generate_custom_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=INP_SIZE))
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(2, activation='softmax'))
    return model


# Function to generate pretrained models
def generate_pretrained_model(model_name):
    if model_name == 'VGG16':
        model = VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=Input(shape=INP_SIZE)
        )
    elif model_name == 'ResNet101':
        model = ResNet101(
            include_top=False,
            weights='imagenet',
            input_tensor=Input(shape=INP_SIZE)
        )
    elif model_name == 'Xception':
        model = Xception(
            include_top=False,
            weights='imagenet',
            input_tensor=Input(shape=INP_SIZE)
        )
    else:
        model = None
        print('Invalid Choice!')
    return model


# Fit model with model name printed in each epoch
def fit_model(model, model_name, x_train, y_train, x_test, y_test):
    optim = Adam(learning_rate=LR)

    if model_name == 'Custom':
        model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])
        print(f"Training {model_name} for {EPOCHS} epochs...")
        history = model.fit(
            trainAug.flow(x_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=len(x_train) // BATCH_SIZE,
            validation_data=(x_test, y_test),
            validation_steps=len(x_test) // BATCH_SIZE,
            epochs=EPOCHS  # Train for all epochs in one call
        )

    else:
        # Freeze base model layers for pre-trained models
        for layer in model.layers:
            layer.trainable = False
        headModel = model.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(64, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(2, activation="softmax")(headModel)
        model = Model(inputs=model.input, outputs=headModel)

        model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])
        print(f"Training {model_name} for {EPOCHS} epochs...")
        history = model.fit(
            trainAug.flow(x_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=len(x_train) // BATCH_SIZE,
            validation_data=(x_test, y_test),
            validation_steps=len(x_test) // BATCH_SIZE,
            epochs=EPOCHS  # Train for all epochs in one call
        )

    return history, model


# Display training history
# Display training history with smooth curves
def display_history(history_):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting loss
    ax[0].plot(history_.history['loss'], color='b', label="training_loss")
    ax[0].plot(history_.history['val_loss'], color='r', label="validation_loss")
    ax[0].set_title('Training and Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='best')

    # Plotting accuracy
    ax[1].plot(history_.history['accuracy'], color='b', label="training_accuracy")
    ax[1].plot(history_.history['val_accuracy'], color='r', label="validation_accuracy")
    ax[1].set_title('Training and Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend(loc='best')

    # Display plots
    plt.tight_layout()
    plt.show()



# Plot confusion matrix and classification report with titles
def plot_metrics(model_, x_test, y_test, model_name):
    plt.figure()
    ax = plt.subplot()
    ax.set_title(f'{model_name} Confusion Matrix')
    pred = model_.predict(x_test, batch_size=BATCH_SIZE)
    pred = np.argmax(pred, axis=1)
    cm = confusion_matrix(np.argmax(y_test, axis=1), pred)
    sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes, cmap='Reds')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
    print(classification_report(np.argmax(y_test, axis=1), pred))
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print("ACC: {:.4f}".format(acc))
    print("Sensitivity: {:.4f}".format(sensitivity))
    print("Specificity: {:.4f}".format(specificity))


print("Training Custom Model...")
# Generate custom model
custom_mod = generate_custom_model()

# Generate pretrained models
vgg_mod = generate_pretrained_model('VGG16')
resnet_mod = generate_pretrained_model('ResNet101')
xception_mod = generate_pretrained_model('Xception')

# Fit and evaluate models
(cus_his, custom_mod) = fit_model(custom_mod, 'Custom', x_train, y_train, x_test, y_test)
display_history(cus_his)
plot_metrics(custom_mod, x_test, y_test, 'Custom')
print("Custom Model Training Completed.")
print("\nTraining VGG16 Model...")
(vgg_his, vgg_mod) = fit_model(vgg_mod, 'VGG16', x_train, y_train, x_test, y_test)
display_history(vgg_his)
plot_metrics(vgg_mod, x_test, y_test, 'VGG16')
print("VGG16 Model Training Completed.")
# ResNet101 Model
print("\nTraining ResNet101 Model...")
(res_his, resnet_mod) = fit_model(resnet_mod, 'ResNet101', x_train, y_train, x_test, y_test)
display_history(res_his)
plot_metrics(resnet_mod, x_test, y_test, 'ResNet101')
print("ResNet101 Model Training Completed.")

# Xception Model
print("\nTraining Xception Model...")
(xcp_his, xception_mod) = fit_model(xception_mod, 'Xception', x_train, y_train, x_test, y_test)
display_history(xcp_his)
plot_metrics(xception_mod, x_test, y_test, 'Xception')
print("Xception Model Training Completed.")

history, model = fit_model(custom_mod, 'Custom', x_train, y_train, x_test, y_test)
display_history(history)
# Ensemble predictions (implementing a simple average)
# Ensemble predictions including Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Function to extract features using a pre-trained model
def extract_features(model, x_data):
    # Remove final layers of the pre-trained model
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    # Extract features
    features = feature_model.predict(x_data, batch_size=BATCH_SIZE)

    # Flatten the features if necessary (depends on the model architecture)
    features = features.reshape((features.shape[0], -1))

    return features


# Extract features from pre-trained models (e.g., VGG16)
vgg_features_train = extract_features(vgg_mod, x_train)
vgg_features_test = extract_features(vgg_mod, x_test)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(vgg_features_train, np.argmax(y_train, axis=1))

# Predict using Random Forest on test data
rf_preds = rf_classifier.predict(vgg_features_test)




def ensemble_with_rf(models, rf_classifier, x_test, rf_features_test):
    # Neural network predictions
    nn_preds = np.mean([model.predict(x_test) for model in models], axis=0)

    # Random Forest predictions (one-hot encoding)
    rf_preds = rf_classifier.predict_proba(rf_features_test)

    # Combine predictions (e.g., simple average)
    final_preds = np.mean([nn_preds, rf_preds], axis=0)

    return final_preds


# Extract test features for RF
rf_features_test = extract_features(vgg_mod, x_test)

# Get final ensemble predictions
ensemble_preds_with_rf = ensemble_with_rf([custom_mod, vgg_mod, resnet_mod, xception_mod], rf_classifier, x_test,
                                          rf_features_test)

# Convert predictions to class labels and evaluate ensemble with RF
ensemble_class_preds_with_rf = np.argmax(ensemble_preds_with_rf, axis=1)
cm_ensemble_rf = confusion_matrix(np.argmax(y_test, axis=1), ensemble_class_preds_with_rf)
sns.heatmap(cm_ensemble_rf, annot=True, xticklabels=classes, yticklabels=classes, cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Ensemble with Random Forest Confusion Matrix')
plt.show()

print(classification_report(np.argmax(y_test, axis=1), ensemble_class_preds_with_rf))
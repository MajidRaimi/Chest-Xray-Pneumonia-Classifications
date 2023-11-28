# Chest X-Ray Image Classification 🌐🔍

Welcome to the Chest X-Ray Image Classification project! This project focuses on classifying chest X-ray images into two categories: "Normal" and "Pneumonia" using a Convolutional Neural Network (CNN) model.

## Project Structure 📁

```
- data/
      - images/
             - normal/
             - pneumonia/
- env/
- .gitignore
- logs/
- models/
- main.ipynb
- README.md
- requirements.txt
```

The `data` directory contains two subdirectories, `normal` and `pneumonia`, each containing a large number of chest X-ray images for training and testing the model.

## Model Architecture 🧠

```python
model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

This CNN architecture consists of three convolutional layers with max-pooling, followed by a flattening layer and two dense layers. The final layer uses the sigmoid activation function for binary classification.

## Dataset 📷

**Chest X-Ray Images (Pneumonia)**  
[Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/)

The `normal` and `pneumonia` folders within the `data/images/` directory contain a substantial number of chest X-ray images for both normal and pneumonia cases.

## Installing Dependencies 🛠️

To set up the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage 🚀

1. Open `main.ipynb` to explore the data, preprocess it, and train the model.
2. The model is trained using TensorFlow. Ensure you have GPU support for faster training.

## Training the Model 🏋️‍♂️

Adjust the number of epochs and other parameters as needed:

```python
history = model.fit(train, epochs=24, validation_data=val, callbacks=[tensorboard_callback])
```

## Model Evaluation 📊

After training, the model's precision, recall, and accuracy can be evaluated using the test dataset.

```python
print('Precision: ', pre.result().numpy())
print('Recall: ', rec.result().numpy())
print('Accuracy: ', acc.result().numpy())
```

## Testing the Model 🧪

Test the model by using the provided `predict` function:

```python
predict('data/images/normal/IM-0001-0001.jpeg')  # Output: Normal
predict('data/images/pneumonia/person1_bacteria_1.jpeg')  # Output: Pneumonia
```

## Save and Load Model ⬇️

Save the trained model for future use:

```python
model.save('models/main_model.keras')
```

Load the saved model:

```python
new_model = load_model('models/main_model.keras')
```

## Contributors 👥

- Majid Saleh Al-Raimi
- Mahmoud Sahal Noor
- Rashid Sami Al-Binali
- Abdulrahman Sami Al-Juhani
- Mashari Adel Al-Jiban


Feel free to customize and expand upon this project. Happy coding! 🚀🤖
# Chest X-Ray Image Classification 🌐🔍

Welcome to the Chest X-Ray Image Classification project! This endeavor represents the culmination of our exploration into the realm of Deep Learning within the context of the **ARTI 502** course. As a final project for this course, our objective is to employ a Convolutional Neural Network (CNN) model to effectively classify chest X-ray images into two distinct categories: "Normal" and "Pneumonia." This undertaking not only showcases our understanding of deep learning principles but also demonstrates the practical application of these concepts in the domain of medical image analysis.

Join us on this journey as we leverage advanced techniques to enhance diagnostic capabilities, contributing to the intersection of artificial intelligence and healthcare. Together, let's unravel the potential of deep learning in accurately identifying health conditions through the analysis of chest X-ray images.
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


## Report 📄
You can find the final report for this project [here](./report.md).

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

## Dataset 🖼️

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
- Alwaleed Ahmad Al-Qurashi
- Mahmoud Sahal Noor
- Rashid Sami Al-Binali
- Abdulrahman Sami Al-Juhani
- Mashari Adel Al-Jiban

## Hardware Specifications 💻
- **Processor:** Apple M2 chip
- **Memory:** 16 GB RAM
- **Storage:** 512 GB SSD
- **Operating System:** macOS Sonoma 14.1.1


## License 📝
You can fine the license for this project [here](./LICENSE).
## License 📄

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

Feel free to customize and expand upon this project. Happy coding! 🚀🤖

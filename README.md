# Digit-Recognition-Using-Deep-Learning
 Utilizing MNIST dataset's 60k training and 10k test samples of 28x28 grayscale images, this project employs DNNs for accurate handwritten digit recognition. With balanced digit classes, it aims to build a robust model without data cleaning, focusing on achieving high accuracy.

 Architecture:
 
The CNN architecture consists of Conv2D layers for feature extraction, with 32 filters in the first layer and 64 in the second, both using a (3, 3) kernel size and ReLU activation. MaxPooling2D layers reduce spatial dimensions using (2, 2) pooling. Dropout layers with rates of 0.25 and 0.5 prevent overfitting. The Flatten layer transforms the output into a 1D vector. Dense layers perform classification, with 256 neurons and ReLU activation in the first layer and 10 neurons with softmax activation in the output layer for class probabilities.

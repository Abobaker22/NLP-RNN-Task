# NLP-RNN-Task
This project trains a machine learning model to classify movie reviews as either **positive** or **negative** using a **Recurrent Neural Network (RNN).**

## Step-by-Step Explanation

### Import Necessary Libraries
- We use **TensorFlow** and **NumPy** for deep learning and numerical operations.
- The dataset comes from **IMDb**, a collection of **50,000 movie reviews** labeled as **positive (1) or negative (0).**

### Load and Preprocess the Data
- We load only the **10,000 most common words** in reviews to keep things manageable (`num_words=10000`).
- Reviews have different lengths, so we **pad** them to a fixed length (`maxlen=200`) to make them uniform.

### Build the RNN Model
- **Embedding Layer:** Converts words into numerical vectors for the model to understand.
- **SimpleRNN Layer:** A basic recurrent layer that helps understand the sequence of words.
- **Dense (Fully Connected) Layer:** Outputs **1 value (between 0 and 1)**, predicting whether a review is **negative (0) or positive (1).**

### Compile the Model
- **Loss Function:** Since this is binary classification (0 or 1), we use `binary_crossentropy` to measure error.
- **Optimizer:** We use `adam` to adjust weights for better accuracy.
- **Metric:** We track `accuracy` to measure how well the model is performing.

### Train the Model
- We train for **5 rounds (epochs)** with **64 reviews per batch**, checking performance on test data after each epoch.

### Evaluate the Model
- After training, we test the model on unseen reviews and print the **accuracy**.

## What Happens Behind the Scenes?
- The model reads each review **word by word** in sequence.
- It **remembers** important words using the **RNN layer**.
- After learning patterns, it predicts whether a new review is **positive or negative**.

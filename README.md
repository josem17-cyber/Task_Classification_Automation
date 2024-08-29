# Task Classification Automation

This project automates the classification of tasks using deep learning. It consists of data preparation, model training, and a web application for making predictions.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Openpyxl
- Numpy

## Project Structure

```plaintext
├── data_preparation.py
├── model.py
├── app.py
├── data
│   └── Automatizacion_Clasificacion_EDA_HIDRAL.xlsx
├── model.h5
└── README.md
```

- `data_preparation.py`: Prepares and processes the data for model training.
- `model.py`: Defines and trains the deep learning model.
- `app.py`: Streamlit web application for making predictions.
- `data/Automatizacion_Clasificacion_EDA_HIDRAL.xlsx`: Dataset file.
- `model.h5`: Saved trained model.
- `README.md`: Project documentation.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/task-classification-automation.git
   cd task-classification-automation
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset file (`Automatizacion_Clasificacion_EDA_HIDRAL.xlsx`) in the `data` directory.

## Usage

### Data Preparation

Run the `data_preparation.py` script to preprocess the data:

```bash
python data_preparation.py
```

### Model

Train the model using the `model.py` script:

```bash
python model.py
```

### Web Application

Start the Streamlit web application to make predictions:

```bash
streamlit run app.py
```

Open your browser and go to `http://localhost:8501` to use the application.

## Model Training

The model is a neural network with the following architecture:
- Input layer
- Dense layers with ReLU activation
- Dropout layers for regularization
- Two output layers for predicting `Categoria` and `Dueño` with softmax activation

Early stopping is used to prevent overfitting.

## Web Application

The web application is built with Streamlit. Users can input a task subject, and the application will predict the `Categoria` and `Dueño` based on the trained model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

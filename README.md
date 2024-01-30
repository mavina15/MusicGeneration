# Music Generation using LSTM

## Introduction
This project utilizes a Long Short-Term Memory (LSTM) neural network to generate new musical sequences based on a dataset of MIDI files. The model is trained to predict the pitch, step, and duration of the next note in a sequence. The training data is sourced from the MAESTRO dataset, and the generated output is saved as a MIDI file.

## Dependencies
Ensure the following dependencies are installed using the provided commands:
```bash
# Install fluidsynth
!sudo apt install -y fluidsynth

# Install Python packages
!pip install --upgrade pyfluidsynth
!pip install pretty_midi
```

## Libraries Used
- **Collections**: Used for handling collections of notes during MIDI file parsing.
- **Datetime**: Provides functionality to work with dates and times.
- **Fluidsynth**: Utilized for synthesizing MIDI files into audio waveforms.
- **Glob**: Used for file path matching using wildcard patterns.
- **Numpy**: Essential for numerical operations and data manipulation.
- **Pathlib**: Simplifies file path manipulation.
- **Pandas**: Used for data manipulation and analysis.
- **Pretty_MIDI**: A library for handling MIDI files in a human-readable format.
- **Seaborn**: A statistical data visualization library based on Matplotlib.
- **Tensorflow**: An open-source machine learning library.

## Data Collection
The project downloads and extracts MIDI files from the MAESTRO dataset if they don't already exist locally.

## MIDI File Analysis
### Part 1
- Loads a sample MIDI file.
- Defines a function to play the MIDI file as audio.
- Displays a piano roll representation of the first 100 notes in the MIDI file.

### Part 2
- Defines functions to convert MIDI files to a DataFrame of note information and vice versa.
- Displays the pitch information of the first few notes in the DataFrame.
- Shows the piano roll representation of the notes.

### Part 3
- Parses multiple MIDI files, concatenates the note information, and creates a TensorFlow dataset.
- Defines a function to create sequences from the dataset for training.
- Sets up the training dataset using TensorFlow.

## Model Architecture
- Uses an LSTM layer with 128 units to predict pitch, step, and duration of the next note.
- Custom loss function (`mse_with_positive_pressure`) penalizes negative predictions.
- Compiles the model with Adam optimizer and sets the loss weights for each output.

## Model Training
- Trains the model on the prepared dataset with early stopping and model checkpoint callbacks.
- Plots the training loss over epochs.

## Music Generation
- Defines a function to predict the next note based on a given sequence.
- Generates a sequence of 120 notes using the trained model.
- Converts the generated notes back to a MIDI file.
- Displays and plays the generated MIDI file.

## Conclusion
This project demonstrates the process of training an LSTM model to generate music based on MIDI files. Experiment with different hyperparameters, model architectures, and datasets to create unique musical compositions.

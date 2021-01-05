# Melody Generator using Tensorlow and Music21
This app will randomly generate a melody based on piano midi files.

## Set up
1. After cloning this repository, create new conda enviroment and install required packages

```
conda create --name myenv
conda activate myenv
pip install -r requirement.txt
```
2. How to use
- If you want train this model, put your midi files into midi_data folder and run train.py file (I recommend you should run model on a computer or VM which has GPU.). After training, run generative.py and it will create a midi file for you.
- If you don't want train, you could use my results. Please download `note_data` and `weights` folders [HERE (Google Drive)](https://drive.google.com/drive/folders/1jCspNEc2wVNa-RnZWRzOvglqWOcatbsy?usp=sharing) and put them in this repo directory, then edit file path in generate.py and helpers.py

```
# generate.py
# #load the notes used to train the model
    with open('note_data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)
# helpers.py
# # Load the weight file in weights folder to each node
    model.load_weights('')

```
3. Enjoy your result

## References
- [How to Generate Music using a LSTM Neural Network in Keras](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)


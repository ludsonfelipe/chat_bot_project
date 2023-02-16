from data_cleaning_preprocessing.cleaning_data import TextCleaning
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import json

def get_encoded_words(string, word2idx):
    """
    Encodes a string into a list of integers, where each integer represents a word
    
    Args:
        string (str): The input string to encode
        word2idx (dict): A dictionary that maps words to their corresponding integer codes
    
    Returns:
        list of int: A list of integers that represent the words in the input string
    """
    encoded_words = []
    for word in string.split():
        if word in word2idx:
            encoded_words.append(word2idx[word])
    return encoded_words

import sys
class AskToChat:
    def __init__(self):
        """
        Initializes the ChatBot object by loading the necessary files and models.
        """
        # Load the word2idx dictionary
        print(sys.path)
        with open('training_model/word2idx.json', 'r') as f:
            self.word2idx = json.load(f)
        
        # Load the idx2word dictionary
        with open('training_model/idx2word.json', 'r') as f:
            self.idx2word = json.load(f)
        
        # Load the chatbot model
        self.model = load_model('training_model/chatbot_model.h5')

    def decode_to_string(self, predict):
        """
        Decodes a prediction from the model into a string of words.
        
        Args:
            predict (numpy.ndarray): A prediction from the chatbot model
        
        Returns:
            str: A string of words that represents the prediction
        """
        keys = [i for i in np.argmax(predict, axis=-1)]
        words = []
        for item in keys[0]:
            words.append(self.idx2word[str(item)])
        return ' '.join(words)

    def chat_answer(self, text):
        """
        Gets a response from the chatbot based on the input text.
        
        Args:
            text (str): The input text to get a response for
        
        Returns:
            str: The response from the chatbot
        """
        # Clean the input text
        text = TextCleaning().clean_text(text=text)
        # Encode the input text into a list of integers
        encoded_words = get_encoded_words(text, self.word2idx)

        # If the input text contains only unknown words, return an error message
        if not encoded_words:
            return 'Sorry, I do not understand what you are saying'
        # Pad the encoded text so that it has a consistent length
        encoded_text = pad_sequences([encoded_words], maxlen=10, padding='post')

        # Get the prediction from the chatbot model ##
        predict = self.model.predict([encoded_text, encoded_text])

        return self.decode_to_string(predict)

import re 
import nltk 

class TextCleaning:
    """
    A class to clean and preprocess text data.
    """
    def __init__(self):
        """
        Initialize the lemmatizer object.
        """
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Clean the text by replacing contractions, removing punctuations, converting to lowercase and stripping whitespaces.
        
        Parameters:
        text (str): The input text to be cleaned.
        
        Returns:
        str: The cleaned text.
        """
        text = text.lower()
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", "would", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"let's", 'let is', text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
        text = text.strip()
        return text

    def lemmatize(self, text):
        """
        Lemmatize the text using the WordNetLemmatizer.
        
        Parameters:
        text (str): The input text to be lemmatized.
        
        Returns:
        str: The lemmatized text.

        """
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(tokens)

    def clean_data(self, data, position=-1):
        """
        Clean and preprocess the data by applying the clean_text() and lemmatize() functions.
        
        Parameters:
        data (list): The input data to be cleaned and preprocessed.
        position (int, optional): The position of the text in the data. Default is -1.
        
        Returns:
        list: The cleaned and preprocessed data.
        """
        cleaned_data = []

        for row in data:
            text = self.clean_text(row[position])
            #text = self.lemmatize(text)
            cleaned_data.append(text)

        return cleaned_data
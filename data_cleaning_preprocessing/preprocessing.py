import ast
import tensorflow as tf
from collections import Counter
import json


class DialogueOrganizer:
    """
    Class for organizing dialogues in a 2D array into a list of tuples, where each tuple represents a dialogue pair.

    Attributes:
        conversations (list): 2D array of dialogues, where each subarray contains information about a dialogue line.
    """
    def __init__(self, conversations):
        self.conversations = conversations
    
    def organize(self, position_sentence=3):
        """
        Organizes the dialogues into a list of tuples, where each tuple represents a dialogue pair.

        Returns:
            list: List of tuples, where each tuple represents a dialogue pair.
        """
        question = [] 
        answer = []
        for cod_line in self.conversations:
            sentences = ast.literal_eval(cod_line[position_sentence])
            for i in range(len(sentences)-1):
                question.append(sentences[i])
                answer.append(sentences[i+1])

        return list(zip(question,answer))



class PreprocessToTf:
    """
    This class takes in a pandas dataframe and preprocesses it for use in a Tensorflow model.
    
    Attributes:
        df (pandas.DataFrame): The dataframe to be preprocessed
        maxlen_question (int): The maximum length of questions
        maxlen_answer (int): The maximum length of answers

    """
    def __init__(self, df, maxlen_question=10, maxlen_answer=10):
        """
        The constructor for the PreprocessToTf class.

        Parameters:
            df (pandas.DataFrame): The dataframe to be preprocessed
            maxlen_question (int): The maximum length of questions (default: 10)
            maxlen_answer (int): The maximum length of answers (default: 10)
        """
        self.df = df
        self.maxlen_question = maxlen_question
        self.maxlen_answer = maxlen_answer

    def replace_unique_words(self, df, col_question='question',col_answer='answer'):
        
        """
        Replace unique words in the question and answer columns with '<OUT>'
        
        Parameters:
            col_question (str): The name of the question column in the dataframe (default: 'question')
            col_answer (str): The name of the answer column in the dataframe (default: 'answer')
        
        Returns:
            df (pandas.DataFrame): The dataframe with unique words in the question and answer columns replaced with '<OUT>'
        """

        # Concatenate the two columns of sentences into one
        df['sentences'] = df[col_question] + ' ' + df[col_answer]

        # Join all sentences into one long string
        all_sentences = ' '.join(df['sentences'].tolist())

        # Split the string into a list of words
        all_words = all_sentences.split()

        # Use the Counter object to count the unique words
        word_counts = Counter(all_words)

        # Create a list of the words that appear only once
        unique_words = [word for word, count in word_counts.items() if count == 1]

        for word in unique_words:
            try:
                df[col_question] = df[col_question].str.replace(f'\b{word}\b', '<OUT>')
                df[col_answer] = df[col_answer].str.replace(f'\b{word}\b', '<OUT>')
                c+=1
            except:
                continue
        return df

    def create_vocab(self, questions, answers):
        
        """
        Create the vocabulary of unique words in the questions and answers
        
        Parameters:
            questions (list): List of questions in the dataframe
            answers (list): List of answers in the dataframe
        
        Returns:
            word2idx (dict): A dictionary mapping words to their indices in the vocabulary
            idx2word (dict): A dictionary mapping indices to their words in the vocabulary
        """

        # Create a vocabulary of unique words
        vocab = set()
        for question, answer in zip(questions, answers):
            vocab.update(question.split())
            vocab.update(answer.split())
        vocab = list(vocab)

        # Add the <START> and <END> tokens to the vocabulary
        vocab.append('<START>')
        vocab.append('<END>')

        # Create word to index and index to word dictionaries
        word2idx = {word: i for i, word in enumerate(vocab)}
        idx2word = {i: word for i, word in enumerate(vocab)}
        
        # Save the Key-values
        with open('NLP Chat Bot/training_model/word2idx.json', 'w') as f:
            json.dump(word2idx, f)

        with open('NLP Chat Bot/training_model/idx2word.json', 'w') as f:
            json.dump(idx2word, f)

        return word2idx, idx2word

    def encode_sentence(self, answers, questions, word2idx, idx2word):
        """
        Encodes the questions and answers as sequences of integers, where each integer corresponds to a word in the vocabulary.

        Parameters:
        answers (list of str): A list of answers to the questions.
        questions (list of str): A list of questions.
        word2idx (dict): A dictionary that maps words to their corresponding index in the vocabulary.
        idx2word (dict): A dictionary that maps indices to their corresponding words in the vocabulary.

        Returns:
        tuple: A tuple of two lists of integers, where the first list represents the encoded questions and the second list represents the encoded answers.
        """
        # Encode the questions
        encoded_questions = [[word2idx[word] for word in question.split()] for question in questions]
        
        # Encode the answers
        encoded_answers = [[word2idx['<START>']] + [word2idx[word] for word in answer.split()] + [word2idx['<END>']] for answer in answers]
        
        return encoded_questions, encoded_answers

    def preprocess(self, col1='question', col2='answer'):
        """
        Preprocesses the questions and answers data by encoding them, creating the vocabulary, and padding them.

        Returns:
        tuple: A tuple of four elements:
            - encoded_questions (list of list of int): The encoded questions.
            - encoded_answers (list of list of int): The encoded answers.
            - word2idx (dict): The word-to-index mapping in the vocabulary.
            - idx2word (dict): The index-to-word mapping in the vocabulary.
        """

        # Replace unique words to <OUT>
        #self.df = self.replace_unique_words(df=self.df)

        # Create lists for questions and answers
        questions = self.df[col1].tolist()
        answers = self.df[col2].tolist()

        # Create the vocabulary
        word2idx, idx2word = self.create_vocab(questions,answers)

        # Encode the sentences in questions and answers
        encoded_questions, encoded_answers = self.encode_sentence(answers, questions, word2idx, idx2word)

        # Padding the questions and answers
        encoded_answers = tf.keras.preprocessing.sequence.pad_sequences(encoded_answers, maxlen=self.maxlen_answer, padding='post')
        encoded_questions = tf.keras.preprocessing.sequence.pad_sequences(encoded_questions, maxlen=self.maxlen_question, padding='post')

        return encoded_questions, encoded_answers, word2idx, idx2word


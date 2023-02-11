from tensorflow.keras.callbacks import EarlyStopping
from data_loader import ImportData
from model import Seq2SeqModel
from preprocessing import TextCleaning, DialogueOrganizer, PreprocessToTf
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# Path of the txts data
movie_lines_url = r'C:\Users\felip\2023\NLP Chat Bot\data\chatbot_dataset\movie_lines.txt'
movie_conversations_url = r'C:\Users\felip\2023\NLP Chat Bot\data\chatbot_dataset\movie_conversations.txt'

# Read the data from the URL
sep = ' +++$+++ '
conversations = ImportData(movie_conversations_url).read_lines_txt(sep = sep)
lines = ImportData(movie_lines_url).read_lines_txt(sep = sep)
print(conversations[0],lines[0])

# Organize the code of conversations in the Dataframe conversation
conversations_codes = DialogueOrganizer(conversations).organize()
df_with_question_and_answers_ids = pd.DataFrame(conversations_codes, columns=['question','answer'])

# Clean the sentences and organize code of every sentences in the lines Dataframe
sentences = TextCleaning().clean_data(lines)
cod_sentence = [sublist[0] for sublist in lines]
list_of_cods_and_sentences = zip(cod_sentence,sentences)
df_cods_and_sentences = pd.DataFrame(list_of_cods_and_sentences, columns=['cod','sentence'])

# Merge both of dataframe organizing the sentences by code
df_question = df_with_question_and_answers_ids.merge(df_cods_and_sentences, left_on='question', right_on='cod', how='left')
df_sentences = df_question.merge(df_cods_and_sentences, left_on='answer', right_on='cod', how='left', suffixes=('_question','_answer'))
df = df_sentences[['sentence_question','sentence_answer']]
df.columns = ['question','answer']

df = df.sample(20000)

#df = replace_unique_words(df)

# Preprocess dataframe to be readable to TensorFlow
preprocessing_to_tensorflow = PreprocessToTf(df)
encoded_questions, encoded_answers, word2idx, idx2word = preprocessing_to_tensorflow.preprocess()

print(encoded_questions)

# Define the proportion of data to use for validation
validation_proportion = 0.3

# Split the data into training and validation sets
question_train, question_val, answer_train, answer_val = train_test_split(encoded_questions, encoded_answers, test_size=validation_proportion, shuffle=True)

# Create the vocabulary lenght variable and others parameters
vocab_size = len(word2idx)
max_question_len = max([len(q) for q in encoded_questions])
max_answer_len = max([len(a) for a in encoded_answers])

# Recieve the Seq2Seq Class, and input the parameters
model = Seq2SeqModel(vocab_size, embedding_size=max_question_len, hidden_size=128)

# Create the model
model = model.create_model()

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Fitting the model
model.fit([question_train, answer_train], 
        answer_train,
          batch_size=64,
          epochs=1,  
          validation_data=([question_val, answer_val],answer_val),  
          callbacks=[early_stopping])
 
# Save the model
model.save('chatbot_model.h5')
#If this file is accessed without the yelp_academic_dataset_review.json file that should be included in the project, then
#this script requires the yelp review dataset, which can be downloaded here: https://www.yelp.com/dataset/download

import os
import pandas as pd
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import FeatureUnion

import nltk
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

#Set seed to make results easier to replicate/adjust
torch.manual_seed(0)

#Stop words (function words) from NLTK
stop_words = set(stopwords.words('english'))

#File path to the original json object, adjust this path or you can comment out this line if you already have the dataset to run the script
#file_path = 'course project\yelp_academic_dataset_review.json'

#To set up the workable dataset from the json file, you can adjust how large you want the dataset to be with total_dataset_length.
#You can set aside a "fake" test dataset to make predictions that is completely set aside from the training set, and set the size
#with test_dataset_length.
total_dataset_length = 20000
test_dataset_length = 4000

#Function to convert Yelp 1-5 stars to a sentiment rating of negative, neutral, or positive
def to_sentiment(stars):
    if stars <= 2:
        return 0  #Negative
    elif stars == 3:
        return 1  #Neutral
    else:
        return 2  #Positive

#Function to process the data in chunks. Full json file was too large to run on my machine without chunking
#max_rows is the number of rows to work with (since original json file is quite large)
#to use whole file, update to not include max_rows but keep chunk_size for chunking out the full json.
def process_data(file_path, chunk_size=10000, max_rows=80000):
    def read_json_in_chunks(file_path, chunk_size):
        chunk_iter = pd.read_json(file_path, lines=True, chunksize=chunk_size)
        for chunk in chunk_iter:
            chunk = chunk[['review_id', 'text', 'stars']]
            chunk['sentiment'] = chunk['stars'].apply(to_sentiment)
            yield chunk
    row_count = 0
    for chunk in read_json_in_chunks(file_path, chunk_size):
        remaining_rows = max_rows - row_count
        if remaining_rows <= 0:
            break
        if len(chunk) > remaining_rows:
            chunk = chunk.iloc[:remaining_rows]

        chunk.to_csv('processed_chunk.csv', mode='a', index=False, header=not os.path.exists('processed_chunk.csv'))
        row_count += len(chunk)

#Process the data in the json file. Comment out this line if you want to use the existing dataset included in this project
#process_data(file_path)

#Relevant file path for the processed chunk. Update if needed.
chunk_file_path = 'course project\processed_chunk.csv'
df1 = pd.read_csv(chunk_file_path)

#Run this to make sure that the CSV was properly created/read
with open(chunk_file_path, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    row_count = 0 
    for row in csvreader:
        print(row)
        row_count += 1
        #row_count is the number of rows to print out
        if row_count == 2:
            #break after printing out n number of rows
            break 

#Load the data in CSV into a pandas dataframe
df = pd.read_csv(chunk_file_path)

#Function to pre-process the text by removing stopwords after converting words to lowercase
def process_text(text):
    words = [word.lower() for word in text.split() if word.lower() not in stop_words]
    return ' '.join(words)

#Function to extract features from the texts
def extract_features(text):
    processed_text = ' '.join([word.lower() for word in text.split() if word.lower() not in stop_words])
    sia = SentimentIntensityAnalyzer()
    sentiment_dict = sia.polarity_scores(processed_text)
    compound_score = sentiment_dict['compound']  # Use the compound score
    return compound_score

#Set a fixed random seed for reproducibility
np.random.seed(42)

#Shuffle the DataFrame with the same random state to get 'random' samples
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

#Split df_shuffled into a fake test set and training set
#First, create fake data (only 'review_id' and 'text' columns) - DO NOT LOOK AT SENTIMENT FOR THESE TEXTS!!!!
fake_test_data = df_shuffled[['review_id', 'text']].head(test_dataset_length)

#Rest of the data is the training set
train_data = df_shuffled.loc[test_dataset_length:total_dataset_length]

#Check for null values in both dfs to make sure it worked correctly.
#null_values_fake_test_data = fake_test_data.isnull().sum()
#null_values_train_data = train_data.isnull().sum()
#print("Null values in fake_test_data:\n", null_values_fake_test_data)
#print("\nNull values in train_data:\n", null_values_train_data)

#Pre-process both dfs
fake_test_data['text'] = fake_test_data['text'].apply(process_text)
train_data['text'] = train_data['text'].apply(process_text)

#Count the frequency of different sentiment values in train_data (for analysis)
#sentiment_counts = train_data['sentiment'].value_counts()
#print("\nSentiment value counts in train_data:\n", sentiment_counts)

#Print statistical information for relevant columns in train_data
#train_data_stats = train_data.describe()
#print("\nStats for train_data:\n", train_data_stats)

#Apply the extract_features function to both datasets
fake_test_data['features_text'] = fake_test_data['text'].apply(extract_features)
train_data.loc[:, 'features_text'] = train_data['text'].apply(extract_features)

#Print statements to make sure everything looks good so far
#print (fake_test_data.head())
#print()
#print(train_data.head())
#print()
#print("Length of fake_test_data:", len(fake_test_data))
#print("Length of train_data:", len(train_data))

#Define the vectorizers for character and word n-grams
char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))

#FeatureUnion to combine character and word n-gram features
combined_features = FeatureUnion([
    ('char_tfidf', char_vectorizer),
    ('word_tfidf', word_vectorizer)
])

#Transform the 'text' column of the train and test data using the combined_features
X_train_text = combined_features.fit_transform(train_data['text'])
X_test_text = combined_features.transform(fake_test_data['text'])

#Convert the features_text to a 2D numpy array with the same number of rows
features_text_array = np.array(train_data['features_text'].tolist()).reshape(-1, 1)

#Combine the vectorized text features with the custom features
X_train = np.hstack((X_train_text.toarray(), features_text_array))
features_text_array_test = np.array(fake_test_data['features_text'].tolist()).reshape(-1, 1)
X_test = np.hstack((X_test_text.toarray(), features_text_array_test))

#Encode labels
y_train = train_data['sentiment'].values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

#Compute the class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
#class_weights[1] *= 1
#class_weights[0] *= 1
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

#Split out the hold out data set for testing. test_size = percentage of the data to hold out
X_temp, X_hold_out, y_temp, y_hold_out = train_test_split(
    X_train, y_train_encoded, test_size=0.35, random_state=42, stratify=y_train_encoded
)

#Split the remaining data into the training and validation set. test_size = percentage of remaining data to be the validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp  # Adjust the test_size as needed
)

#SentimentClassifier class
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, num_classes)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
#Hyperparameters for adjusting
input_size = X_train.shape[1]
hidden_size = 68 
num_classes = 3
learning_rate = 0.001
batch_size = 64 
num_epochs = 20
weight_decay = 0

#Model, loss, and optimizer
model = SentimentClassifier(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#Data loaders for training, validaiton, and hold out sets
train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
val_loader = DataLoader(val_dataset, batch_size=batch_size)

hold_out_dataset = TensorDataset(torch.tensor(X_hold_out).float(), torch.tensor(y_hold_out).long())
hold_out_loader = DataLoader(hold_out_dataset, batch_size=batch_size)

#Early stopping parameters
early_stop = 3
epochs_not_improving = 0
min_val_loss = np.Inf

#Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}")

    #Early stopping check
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        epochs_not_improving = 0
    else:
        epochs_not_improving += 1
        if epochs_not_improving == early_stop:
            print('Epochs not improving - early stop')
            break

#Function to calculate accuracy, get predictions, and save misclassifications
def evaluate_model(data_loader, save_misclassified=True):
    total, correct = 0, 0
    all_labels, all_predictions, misclassified = [], [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

            if save_misclassified:
                mismatches = predicted != labels
                misclassified_data = inputs[mismatches]
                misclassified_labels = labels[mismatches]
                misclassified_preds = predicted[mismatches]
                for input_data, true_label, pred_label in zip(misclassified_data, misclassified_labels, misclassified_preds):
                    misclassified.append({'input_data': input_data, 'true_label': true_label.item(), 'predicted_label': pred_label.item()})

    accuracy = 100 * correct / total
    
    #Save misclassified instances to CSV
    if save_misclassified and misclassified:
        df_misclassified = pd.DataFrame(misclassified)
        df_misclassified.to_csv('misclassified_predictions.csv', index=False)
        print("Misclassified predictions saved to misclassified_predictions.csv")

    return accuracy, all_labels, all_predictions

#Evaluate validation set
validation_accuracy, all_labels_val, all_predictions_val = evaluate_model(val_loader, save_misclassified=False)
print(f'Validation Accuracy: {validation_accuracy}%')
print("Classification Report on Validation Set:\n", classification_report(all_labels_val, all_predictions_val))

#Evaluate hold-out set
hold_out_accuracy, all_labels_hold_out, all_predictions_hold_out = evaluate_model(hold_out_loader, save_misclassified=True)
print(f'Hold Out Set Accuracy: {hold_out_accuracy}%')
print("Classification Report on Hold Out Set:\n", classification_report(all_labels_hold_out, all_predictions_hold_out))

#Make predictions using the test data
test_dataset = TensorDataset(torch.tensor(X_test).float())
test_loader = DataLoader(test_dataset, batch_size=batch_size)

predictions = []
with torch.no_grad():
    for inputs, in test_loader:
        output = model(inputs)
        _, pred = torch.max(output, dim=1)
        predictions.extend(label_encoder.inverse_transform(pred.numpy()))

#Save the predictions to CSV
output_data = fake_test_data.copy()
output_data['sentiment'] = predictions
output_data[['review_id', 'sentiment']].to_csv('sentiment_predictions.csv', index=False)
print("Predictions saved to sentiment_predictions.csv")

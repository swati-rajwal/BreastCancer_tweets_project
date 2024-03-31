import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re, string

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Load and preprocess data
# [Your data loading code here]
df_train = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/Breast Cancer Data/Classification_ data/train.csv')
df_dev = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/Breast Cancer Data/Classification_ data/dev_.csv')
df_test = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/Breast Cancer Data/Classification_ data/test.csv')

print(f"Train set shape: {df_train.shape}")
print(f"Dev set shape: {df_dev.shape}")
print(f"Test set shape: {df_test.shape}")

# Tokenization and data preparation
tokenizer_roberta = RobertaTokenizerFast.from_pretrained("roberta-base")
lemmatizer = WordNetLemmatizer()

def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess_text_optimized(text):
    text = remove_emojis(text)
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('@', '').replace('\n', '')
    tokens = word_tokenize(text)
    filtered = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return ' '.join(filtered)

# Preprocess your text data
df_train['text']= df_train['text'].apply(preprocess_text_optimized)
df_dev['text'] = df_dev['text'].apply(preprocess_text_optimized)
df_test['text'] = df_test['text'].apply(preprocess_text_optimized)

# Convert data to PyTorch tensors
def tokenize_roberta(data, max_len):
    input_ids = []
    attention_masks = []
    for txt in data:
        encoded = tokenizer_roberta.encode_plus(
            txt, add_special_tokens=True, max_length=max_len,
            truncation=True, padding='max_length',
            return_attention_mask=True, return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'][0])
        attention_masks.append(encoded['attention_mask'][0])
    return torch.stack(input_ids), torch.stack(attention_masks)


tokenizer_roberta = RobertaTokenizerFast.from_pretrained("roberta-base")
X_train = df_train['text'].values
y_train = df_train['Class'].values


X_valid = df_dev['text'].values
y_valid = df_dev['Class'].values

X_test = df_test['text'].values
y_test = df_test['Class'].values

token_lens = []
for txt in X_train:
    tokens = tokenizer_roberta.encode(txt, max_length=512, truncation=True)
    token_lens.append(len(tokens))
max_length=np.max(token_lens)
MAX_LEN=max_length
# max_length = 512  # Set your max_length

train_input_ids, train_attention_masks = tokenize_roberta(X_train, max_length)
val_input_ids, val_attention_masks = tokenize_roberta(X_valid, max_length)
test_input_ids, test_attention_masks = tokenize_roberta(X_test, max_length)

# Convert labels to one-hot encoding
ohe = OneHotEncoder()
y_train = torch.tensor(ohe.fit_transform(np.array(y_train).reshape(-1, 1)).toarray())
y_valid = torch.tensor(ohe.transform(np.array(y_valid).reshape(-1, 1)).toarray())
y_test = torch.tensor(ohe.transform(np.array(y_test).reshape(-1, 1)).toarray())

# Create PyTorch datasets and dataloaders
train_data = TensorDataset(train_input_ids, train_attention_masks, y_train)
val_data = TensorDataset(val_input_ids, val_attention_masks, y_valid)
test_data = TensorDataset(test_input_ids, test_attention_masks, y_test)

batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Define the model
class RoBERTaClassifier(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[1]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roberta_model = RobertaModel.from_pretrained('roberta-base')
model = RoBERTaClassifier(roberta_model, 2)
model.to(device)

# Training the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels.argmax(dim=1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

# Evaluate the model
# [Add evaluation and prediction code for PyTorch model here]
# Function to evaluate the model
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels.argmax(dim=1))
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.argmax(dim=1).cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    return avg_loss, all_preds, all_labels

# Evaluate on validation set
val_loss, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
print(f'Validation Loss: {val_loss}')
print('\nValidation Classification Report:\n', classification_report(val_labels, val_preds))
df_dev['roberta_predictions'] = val_preds
df_dev.to_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/Breast Cancer Data/Predictions/dev_with_roberta_preds.csv',index=False)
print("test set predictions saved!",flush=True)
# Predict on test set
test_loss, test_preds, test_labels = evaluate_model(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss}')
print('\nTest Classification Report:\n', classification_report(test_labels, test_preds))
df_test['roberta_predictions'] = test_preds

df_test.to_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/Breast Cancer Data/Predictions/test_with_roberta_preds.csv',index=False)
print("test set predictions saved!",flush=True)
# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
print('\nConfusion Matrix:\n', cm)

# Save the model
torch.save(model.state_dict(), '/local/scratch/shared-directories/ssanet/swati_folder/bionlp/Trained_roberta.pth')

# Load the model
# model.load_state_dict(torch.load('/path/to/save/model.pth'))
# Assuming df_full is your new data loaded from a CSV file

"""
import pandas as pd
df1 = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/Breast Cancer Data/BC_full_data.csv')
len(df1)
1454637
df1 = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/BC_full_data_preprocessed.csv')
len(df1)
1454637
 df1 = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/Breast Cancer Data/BC_full_data.csv')
df2 = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/BC_full_data_preprocessed.csv')
df1.columns
Index(['0', 'gecrginasprks', 'Sun Jun 03 18:32:06 +0000 2018',
       '2021-03-22 21:09:08.449055', '1003343585490583553', 'True', 'Emory',
       '@loveuthe1sttime cancer sun aries moon leo rising',
       '1374105868531200000', 'Unnamed: 9', 'Unnamed: 10',
       'Mon Mar 22 21:09:03 +0000 2021', 'she/they 🇫🇷'],
      dtype='object') df1.iloc[:,7]
0          You tryin bring me home know this is part of t...
1          I really cannot believe they voted out the one...
2          water signs [cancer + scorpio + pisces]-\nyou ...
3                   @mimithesexdeity cancer/capricorn axis 😭
4          Cam's Corner her Battle with Stage 3 Breast Ca...
                                 ...                        
1454632                       So so sad to hear this!  R.I.P
1454633                                             omg non?
1454634    Very sad. Helen McCrory was a wonderful actor....
1454635    Mental Illnesss - SIXTY-NINTH STREET SUICIDE -...
1454636    💔🙏🎗💛 another precious child lost to pediatric ...
Name: @loveuthe1sttime cancer sun aries moon leo rising, Length: 1454637, dtype: object
df2['original_text'] = df1.iloc[:,7]
df2.sample(3)
                                                      text                                      original_text
718355                             super caaaute cancer ♋️              This is super caaaute!!😍 \n#Cancer ♋️
1343788  powerful backer helping reach goal cancer http...  You have powerful backers helping you reach yo...
630949                                        actually cry                               I AM ACTUALLY CRYING
df2.to_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/BC_full_data_preprocessed.csv',index=False)
"""

df_full = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/BC_full_data_preprocessed.csv')
print(f"Columns in df_full preprocessed: {df_full.columns}",flush=True)
# Preprocess and tokenize your new data
# df_full['text'] = df_full['text'].apply(preprocess_text_optimized)
df_full.dropna(inplace=True)
X_full = df_full['text'].values
full_input_ids, full_attention_masks = tokenize_roberta(X_full, MAX_LEN)

# Convert to PyTorch tensors
full_dataset = TensorDataset(full_input_ids, full_attention_masks)
full_loader = DataLoader(full_dataset, batch_size=16)  # You can adjust the batch size

# Use the model to predict
model.eval()
predictions = []

with torch.no_grad():
    for input_ids, attention_mask in full_loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())

# Add predictions to the DataFrame
df_full['roberta_predictions'] = predictions

# Save the DataFrame with predictions
df_full.to_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/Breast Cancer Data/Predictions/BC_full_data_roberta_predictions.csv', index=False)
print("BC_full_data_roberta predictions saved!!",flush=True)


################################ SENTIMENT ANALYSIS ########################################
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from bertopic import BERTopic
import gensim
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import string

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('vader_lexicon')

#### Reading roberta predicted data #######
df = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/Breast Cancer Data/Predictions/BC_full_data_roberta_predictions.csv')
df = df[df['roberta_predictions']==1].copy()
print(f"unique class labels: {df['roberta_predictions'].unique()}")


sia = SentimentIntensityAnalyzer()
df['sentiment_on_original'] = df['original_text'].apply(lambda x: (sia.polarity_scores(x)['compound'] + 1) / 2)   ## Original text
df['sentiment_on_preprocessed'] = df['text'].apply(lambda x: (sia.polarity_scores(x)['compound'] + 1) / 2)    # Preprocessed text
df.to_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/Breast Cancer Data/Predictions/roberta_prediction_1_only_with_sentiment.csv',index=False)
print("Sentment scores file saved!!!")

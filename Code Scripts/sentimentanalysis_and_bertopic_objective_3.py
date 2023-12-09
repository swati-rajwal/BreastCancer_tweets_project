################## BERTOPIC ########################
# df = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/Breast Cancer Data/Predictions/BC_full_data_roberta_predictions_with_sentiment_score.csv')
# df = df[df['roberta_predictions']==1].copy()
from bertopic import BERTopic
import pandas as pd
import nltk
import pandas as pd
import gensim
import nltk
import string
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('vader_lexicon')
df = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/bionlp/Breast Cancer Data/Predictions/roberta_prediction_1_only_with_sentiment.csv')

print("df original shape:", df.shape)
# df = df[df['sentiment_on_preprocessed']<=0.5].copy()
# df = df.reset_index(drop=True)
# print("df shape after only negative sentiments:", df.shape)

treatment_words_2 = ["treatment","medication","medicine","tablets","side effect","reaction","drug","tamoxifen","chemo","mental","emotion",
                     "docetaxel","oncologist","doctor","appointment"]

treatment_regex = r'(?i)\b(?:' + '|'.join(treatment_words_2) + r')\b'
df_filtered = df[df['original_text'].str.contains(treatment_regex, na=False)].copy()
df_filtered = df_filtered.reset_index(drop=True)
print(df_filtered.shape)

self_reported_texts = df['original_text'].to_list()

topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(self_reported_texts)

# topic_model.visualize_barchart(top_n_topics=50)
import matplotlib.pyplot as plt

# Assuming topic_model is your topic modeling object and it has a visualize_barchart method
fig = topic_model.visualize_barchart(top_n_topics=30)
fig.write_html("/local/scratch/shared-directories/ssanet/swati_folder/bionlp/task3_4_test_final.html")
print("HTML save!!", flush=True)
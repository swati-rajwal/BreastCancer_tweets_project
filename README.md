# Unveiling Voices: Identification of Concerns in a Social Media Breast Cancer Cohort via Natural Language Processing

## 📚 Pipeline
<img width="951" alt="image" src="https://github.com/swati-rajwal/BreastCancer_tweets_project/assets/145946818/79afc838-8dbe-4c34-b626-801817c22944">

Figure 1: Natural language processing pipeline for identifying Treatment Discontinuation Topics among breast cancer twitter cohort.

## 🎯Objective
Our primary objectives were threefold:
1. Develop a self-reported breast cancer tweet identification system utilizing traditional Machine learning models and RoBERTa.
2. Identify breast cancer-related concern-based topics in patients’ tweets.
3. Perform sentiment intensity analysis of patients who voice dissatisfaction and identification of treatment discontinuation in the self-reported tweets category.

## 🏃‍♂️To Run the code
1. Python 3 is used as the programming language
2. Download: <a href="https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?resourcekey=0-wjGZdNAUop6WykTtMip30g" target="_blank">GoogleNews-vectors-negative300.bin</a>
3. We have used Jupyter Notebook for most of the coding purposes
4. The sequence of code to run is marked by the alphabet prefix in the file name (A to E)
5. Dataset is available at request

## 📈Results

| Model         | Hyperparameter                  | F1 Micro | F1 Macro | F2 Micro | F2 Macro | Log loss |
|---------------|---------------------------------|----------|----------|----------|----------|----------|
| Decision Tree | criterion='gini', max_depth=10  | 0.778    | 0.608    | 0.778    | 0.596    | 0.734    |
| Logistic Reg. | C=10, penalty='l2'              | 0.772    | 0.576    | 0.772    | 0.570    | 0.464    |
| Naïve Bayes   | alpha=0.1                       | 0.745    | 0.427    | 0.745    | 0.468    | 0.568    |
| Random forest | max_depth=None, n_estimators=50 | 0.752    | 0.476    | 0.752    | 0.498    | 0.652    |
| **RoBERTa**       | epochs=20, batch_size=16        | **0.894**    | 0.853    | 0.894    | 0.841    | 0.332    |

Table 1: Classification Results across various Evaluation Metrics


## 📑 Citation

Please consider citing 📑 our paper if our repository is helpful to your work.
```bibtex
@inproceedings{rajwal-etal-2024-unveiling,
    title = "Unveiling Voices: Identification of Concerns in a Social Media Breast Cancer Cohort via Natural Language Processing",
    author = "Rajwal, Swati  and Pandey, Avinash Kumar  and Han, Zhishuo  and Sarker, Abeed",
    editor = "Demner-Fushman, Dina  and Ananiadou, Sophia  and Thompson, Paul  and Ondov, Brian",
    booktitle = "Proceedings of the First Workshop on Patient-Oriented Language Processing (CL4Health) @ LREC-COLING 2024",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.cl4health-1.32",
    pages = "264--270"
}
```

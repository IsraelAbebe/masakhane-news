import os
import pandas as pd
import tqdm
import numpy as np

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import os
import argparse

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="classification on classical model")
    parser.add_argument("--seed", type=int, default=42, help="Lets use differnt seed")
    
    return parser
    
    
    
def main():
    parser = get_parser()
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    folder_name = '../../data/'

    feature_column = "headline_text"
    label_column = "category"

    language_list = ['amh','eng','fra','hau','ibo','lin','pcm','run','swa','yor','sna','orm']
    print(language_list)
    for language in language_list:
        print('-------------------------------------------------')
        print(f'--------------Working on {language}-----------------')

        train_data = pd.read_csv(f'{folder_name}/{language}/train.tsv',sep='\t')
        dev_data = pd.read_csv(f'{folder_name}/{language}/dev.tsv',sep='\t')
        test_data = pd.read_csv(f'{folder_name}/{language}/test.tsv',sep='\t')
        
        
         # #COMBINE HEADLNE AND TEXT
        train_data["headline_text"] = train_data["headline"].astype(str) + train_data["text"].astype(str)
        dev_data["headline_text"] = dev_data["headline"].astype(str) + dev_data["text"].astype(str)
        test_data["headline_text"] = test_data["headline"].astype(str) + test_data["text"].astype(str)

        print(f' Training set size : {train_data.size}   Dev set size: {dev_data.size}')

        all_text_list  = train_data[feature_column].values.tolist()+dev_data[feature_column].values.tolist() 

        print('[INFO] Sample data \n',all_text_list[:3])
        
        
        
        
        # print(train_data)

        train_text,train_label = train_data[feature_column].values.tolist(),train_data[label_column].values.tolist()
        dev_text,dev_label = dev_data[feature_column].values.tolist(),dev_data[label_column].values.tolist()
        test_text,test_label = test_data[feature_column].values.tolist(),test_data[label_column].values.tolist()

        
        unique_label = train_data[label_column].unique().tolist()

        print('[INFO] Found Labels : ',unique_label)
        # CountVectorizer
        vectorizer = CountVectorizer(analyzer='char_wb',ngram_range=(1, 3))
        vectorizer.fit_transform(all_text_list)

        X_train = vectorizer.transform(train_text).toarray()
        X_dev= vectorizer.transform(dev_text).toarray()
        X_test= vectorizer.transform(test_text).toarray()

        y_train = []
        for i in train_label:
            y_train.append(unique_label.index(i))

        y_dev = []
        for i in dev_label:
            y_dev.append(unique_label.index(i))

        y_test = []
        for i in test_label:
            y_test.append(unique_label.index(i))

        print(f'Sizes : {X_train.shape,X_dev.shape,X_test.shape,len(y_train),len(y_dev),len(y_test)}')



        print('=======   GaussianNB   =========')

        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        # Predict Class
        y_pred = classifier.predict(X_dev)

        # Accuracy 
        accuracy = metrics.accuracy_score(y_dev, y_pred)
        f1 = metrics.f1_score(y_dev, y_pred, average='macro')


        print(f'acc: {accuracy}     |  f1_score: {f1}')
        print(metrics.classification_report(y_dev, y_pred, target_names=unique_label))


        if not os.path.exists(f"GaussianNB/{language}"):
            os.makedirs(f"GaussianNB/{language}")

        
        
        y_pred = classifier.predict(X_test)
        
        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred,average='weighted')
        precision = metrics.precision_score(y_test, y_pred,average='weighted')
        recall = metrics.recall_score(y_test, y_pred,average='weighted')

        print(f"f1 = {f1}")
        print(f"loss = {None}")
        print(f"precision = {precision}")
        print(f"recall = {recall}")


        with open(f"GaussianNB/{language}/test_results{args.seed}.txt", 'w') as f:
            f.write(f"f1 = {f1}\n")
            f.write(f"loss = {None}\n")
            f.write(f"precision = {precision}\n")
            f.write(f"recall = {recall}\n")

        print(f"[INFO] Saved GaussianNB/{language}/test_results{args.seed}.txt")
        f.close()



        print('=======   MultinomialNB   =========')

        classifier = MultinomialNB()
        classifier.fit(X_train, y_train)

        # Predict Class
        y_pred = classifier.predict(X_dev)

        # Accuracy 
        accuracy = metrics.accuracy_score(y_dev, y_pred)
        f1 = metrics.f1_score(y_dev, y_pred, average='macro')


        print(f'acc: {accuracy}     |  f1_score: {f1}')
        print(metrics.classification_report(y_dev, y_pred, target_names=unique_label))
        
        
        # TESTING
        
        if not os.path.exists(f"MultinomialNB/{language}"):
            os.makedirs(f"MultinomialNB/{language}")


        y_pred = classifier.predict(X_test)
        
        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred,average='weighted')
        precision = metrics.precision_score(y_test, y_pred,average='weighted')
        recall = metrics.recall_score(y_test, y_pred,average='weighted')

        print(f"f1 = {f1}")
        print(f"loss = {None}")
        print(f"precision = {precision}")
        print(f"recall = {recall}")


        with open(f"MultinomialNB/{language}/test_results{args.seed}.txt", 'w') as f:
            f.write(f"f1 = {f1}\n")
            f.write(f"loss = {None}\n")
            f.write(f"precision = {precision}\n")
            f.write(f"recall = {recall}\n")

        print(f"[INFO] Saved MultinomialNB/{language}/test_results{args.seed}.txt")
        f.close()


        print('=======   KNeighborsClassifier   =========')

        classifier = KNeighborsClassifier(n_neighbors=3)
        classifier.fit(X_train, y_train)

        # Predict Class
        y_pred = classifier.predict(X_dev)

        # Accuracy 
        accuracy = metrics.accuracy_score(y_dev, y_pred)
        f1 = metrics.f1_score(y_dev, y_pred, average='macro')


        print(f'acc: {accuracy}     |  f1_score: {f1}')
        print(metrics.classification_report(y_dev, y_pred, target_names=unique_label))

        if not os.path.exists(f"KNeighborsClassifier/{language}"):
            os.makedirs(f"KNeighborsClassifier/{language}")

        y_pred = classifier.predict(X_test)
        
        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred,average='weighted')
        precision = metrics.precision_score(y_test, y_pred,average='weighted')
        recall = metrics.recall_score(y_test, y_pred,average='weighted')

        print(f"f1 = {f1}")
        print(f"loss = {None}")
        print(f"precision = {precision}")
        print(f"recall = {recall}")


        with open(f"KNeighborsClassifier/{language}/test_results{args.seed}.txt", 'w') as f:
            f.write(f"f1 = {f1}\n")
            f.write(f"loss = {None}\n")
            f.write(f"precision = {precision}\n")
            f.write(f"recall = {recall}\n")

        print(f"[INFO] Saved KNeighborsClassifier/{language}/test_results{args.seed}.txt")
        f.close()

        print('=======   MLPClassifier   =========')

        classifier = MLPClassifier(random_state=args.seed, max_iter=300)
        classifier.fit(X_train, y_train)

        # Predict Class
        y_pred = classifier.predict(X_dev)

        # Accuracy 
        accuracy = metrics.accuracy_score(y_dev, y_pred)
        f1 = metrics.f1_score(y_dev, y_pred, average='macro')


        print(f'acc: {accuracy}     |  f1_score: {f1}')
        print(metrics.classification_report(y_dev, y_pred, target_names=unique_label))

        if not os.path.exists(f"MLPClassifier/{language}"):
            os.makedirs(f"MLPClassifier/{language}")

        y_pred = classifier.predict(X_test)
        
        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred,average='weighted')
        precision = metrics.precision_score(y_test, y_pred,average='weighted')
        recall = metrics.recall_score(y_test, y_pred,average='weighted')

        print(f"f1 = {f1}")
        print(f"loss = {None}")
        print(f"precision = {precision}")
        print(f"recall = {recall}")


        with open(f"MLPClassifier/{language}/test_results{args.seed}.txt", 'w') as f:
            f.write(f"f1 = {f1}\n")
            f.write(f"loss = {None}\n")
            f.write(f"precision = {precision}\n")
            f.write(f"recall = {recall}\n")

        print(f"[INFO] Saved MLPClassifier/{language}/test_results{args.seed}.txt")
        f.close()

        print('=======   XGBClassifier   =========')

        classifier = XGBClassifier(seed=args.seed)
        classifier.fit(X_train, y_train)

        # Predict Class
        y_pred = classifier.predict(X_dev)

        # Accuracy 
        accuracy = metrics.accuracy_score(y_dev, y_pred)
        f1 = metrics.f1_score(y_dev, y_pred, average='macro')


        print(f'acc: {accuracy}     |  f1_score: {f1}')
        print(metrics.classification_report(y_dev, y_pred, target_names=unique_label))

        if not os.path.exists(f"XGBClassifier/{language}"):
            os.makedirs(f"XGBClassifier/{language}")

        y_pred = classifier.predict(X_test)
        
        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred,average='weighted')
        precision = metrics.precision_score(y_test, y_pred,average='weighted')
        recall = metrics.recall_score(y_test, y_pred,average='weighted')

        print(f"f1 = {f1}")
        print(f"loss = {None}")
        print(f"precision = {precision}")
        print(f"recall = {recall}")


        with open(f"XGBClassifier/{language}/test_results{args.seed}.txt", 'w') as f:
            f.write(f"f1 = {f1}\n")
            f.write(f"loss = {None}\n")
            f.write(f"precision = {precision}\n")
            f.write(f"recall = {recall}\n")

        print(f"[INFO] Saved XGBClassifier/{language}/test_results{args.seed}.txt")
        f.close()


        print('=======   SVC   =========')
        classifier = SVC(gamma='auto',random_state=args.seed)
        classifier.fit(X_train, y_train)
        # Predict Class
        y_pred = classifier.predict(X_dev)

        # Accuracy 
        accuracy = metrics.accuracy_score(y_dev, y_pred)
        f1 = metrics.f1_score(y_dev, y_pred, average='micro')


        print(f'acc: {accuracy}     |  f1_score: {f1}')
        print(metrics.classification_report(y_dev, y_pred, target_names=unique_label))

        if not os.path.exists(f"SVC/{language}"):
            os.makedirs(f"SVC/{language}")

        y_pred = classifier.predict(X_test)
        
        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred,average='weighted')
        precision = metrics.precision_score(y_test, y_pred,average='weighted')
        recall = metrics.recall_score(y_test, y_pred,average='weighted')

        print(f"f1 = {f1}")
        print(f"loss = {None}")
        print(f"precision = {precision}")
        print(f"recall = {recall}")


        with open(f"SVC/{language}/test_results{args.seed}.txt", 'w') as f:
            f.write(f"f1 = {f1}\n")
            f.write(f"loss = {None}\n")
            f.write(f"precision = {precision}\n")
            f.write(f"recall = {recall}\n")

        print(f"[INFO] Saved SVC/{language}/test_results{args.seed}.txt")
        f.close()

        
if __name__ == "__main__":
    main()

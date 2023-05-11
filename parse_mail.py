import os
import email
import pandas as pd
from numpy import full, concatenate
from itertools import chain
from time import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

folder = "pruebas"
datasets = ["easy_ham", "spam_2"]
classes  = [0, 1]
datapath = [os.path.join(folder, dataset, "") for dataset in datasets]
files =  [os.listdir(path) for path in datapath]
number_of_inputs = [len(file) for file in files]
files = [ d+file   for f, d in  zip(files, datapath) for file in f]
Ys = concatenate([full((size,),length) for size, length in zip(number_of_inputs, classes)])

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

class email_iter:
    def __init__(self, files, ):
        self.files=iter(files)
        

    def __iter__(self):
        return self

    def __next__(self):
        file =  self.files.__next__()
        print(file)
        mail = open(file)
        try:
            msg = email.message_from_file(mail)
        except UnicodeDecodeError:
            mail.close()
            mail = open(file)
            payload = ""
            try:
                line = mail.read_line()
                while( line ):
                    payload = payload + line
                    line = mail.read_line()
            except:
                pass
            mail.close()
            return payload

        mail.close()

        payload = msg.get_payload()
        while type(payload) == list:
            payload = payload[0].get_payload()
        #print(payload)
        return payload
                
print("Performing feature extraction...")
Xs = vectorizer.fit_transform(email_iter(files))

df_bow_sklearn = pd.DataFrame(Xs.toarray(), columns= vectorizer.get_feature_names_out())

pipeline = Pipeline(
    [
        ("tfidf", TfidfTransformer()),
        ("clf", SGDClassifier()),
    ]
)
print("...Done")

parameters = {
    
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    "clf__max_iter": (2000,),
    "clf__alpha": (0.00001, 0.000001),
    "clf__penalty": ("l2", "elasticnet"),
    # 'clf__max_iter': (10, 50, 80),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
print(parameters)


t0 = time()
grid_search.fit(Xs, Ys)
print("done in %0.3fs" % (time() - t0))

print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

###############################################PARTE 2####################################################

# carpeta con los correos electrónicos de prueba
prueba_folder = "validaciones"
prueba_files = os.listdir(prueba_folder)

# cargar el modelo entrenado
clf = grid_search.best_estimator_

# extraer características de los correos electrónicos de prueba
X_prueba = vectorizer.transform(email_iter([os.path.join(prueba_folder, f) for f in prueba_files]))
X_prueba_tfidf = TfidfTransformer().fit_transform(X_prueba)

# hacer predicciones sobre si cada correo electrónico es spam o no
predicciones = clf.predict(X_prueba_tfidf)

# inicializar contadores
spam_count = 0
no_spam_count = 0

# imprimir resultados y contar spam y no spam
for f, pred in zip(prueba_files, predicciones):
    if pred == 1:
        print(f"{f}: spam")
        spam_count += 1
    else:
        print(f"{f}: no spam")
        no_spam_count += 1

# imprimir resultados totales
print("Total de correos electrónicos de prueba: ", len(prueba_files))
print("Total de spam: ", spam_count)
print("Total de no spam: ", no_spam_count)


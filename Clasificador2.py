import os
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import os

# Ruta del archivo
file_path = "prueba.txt"

# Ruta a las carpetas con los correos electrónicos de prueba
easy_ham_folder = 'easy_ham'
spam_folder = 'spam_2'

# Función para leer los correos electrónicos y crear una lista de documentos
def read_emails(folder):
    documents = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='latin1') as f:
            text = f.read()
            documents.append(text)
    return documents

# Leer los correos electrónicos de las carpetas easy_ham y spam_2
easy_ham_emails = read_emails(easy_ham_folder)
spam_emails = read_emails(spam_folder)

# Crear un vectorizador de palabras para transformar los documentos en vectores de características
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(easy_ham_emails + spam_emails)
y = np.concatenate((np.zeros(len(easy_ham_emails)), np.ones(len(spam_emails))))

# Separar los datos en un conjunto de entrenamiento y otro de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar un perceptrón en los datos de entrenamiento
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# Evaluar la precisión del modelo en los datos de prueba
accuracy = perceptron.score(X_test, y_test)
print("Accuracy:", accuracy)

# Clasificar un correo electrónico nuevo
# Comprobar si el archivo existe
if not os.path.exists(file_path):
    print("El archivo no existe.")
else:
    # Leer el contenido del archivo
    with open(file_path, "r") as f:
        content = f.read()

    # Vectorizar el contenido del archivo
    new_email_vector = vectorizer.transform([content])

    # Predecir si el correo es spam o no
    prediction = perceptron.predict(new_email_vector)[0]

    # Imprimir el resultado
    if prediction == 1:
        print("Este correo es spam.")
    else:
        print("Este correo no es spam.")

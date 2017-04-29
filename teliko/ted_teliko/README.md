# Ted-First-Assignment
Ted assignment 1


Installing requirements:
```
pip install -r requirements.txt
```
Όλα τα script εχουν σχεδιαστεί για να τρέχουν απο το directory που βρισκονται.

Δημιουργία WordCloud:

	* Εκτελείται απο το αρχείο ted/clouds.py .
	* Χρησιμοποιήθηκε η βιβλιοθήκη wordcloud (https://github.com/amueller/word_cloud) .
	* Υλοποιήθηκε ενα wordcloud για καθε κατηγορία.
	* Τα wordclouds αποθηκεύονται στον φάκελο outputs.
	* Για την δημιουργία των wordclouds χρησιμοποιήθηκε μόνο το Content των άρθρων.

K-Means our version:

	* Εκτελείται απο το αρχείο ted/run_kmeans.py .
	* Δίνοντας στο run_kmeans το argument -p δημιουργούνται τα plots και 		  αποθηκεύονται.
	* Τα δεδομένα επεξεργάζονται με τα παρακάτω εργαλεία:

		Με το όρισμα stop_words='english' αφαιρούνται οι κοινές αγγλικές λέξεις (and, the κτλπ)
		    vectorizer = CountVectorizer(stop_words='english')

		Για να δημιουργηθούν τα plot (και για λόγους ταχύτητας) μπορεί να δοθεί n_components=2. Αλλα για καλύτερο clustering τα 40 components αποδίδουν καλύτερα.
   		    svd = TruncatedSVD(n_components=40)

 		    transformer=TfidfTransformer()

K-Means sklearn version:

    * Έχουμε επίσης και μια υλοποίηση που χρησιμοποιεί τον K-Means αλγόριθμο που προφέρει η βιβλιοθήκη scikit-learn.
      Η υλοποίηση αυτή τρεχει με το σκριπτ skKMeans.py και παράγει ενα επιπλέον csv αρχείο με τα αποτελέσματά του.

Classification:

	* Εκτελείται απο το αρχείο ted/classifications.py.
	* ΒΕΛΤΙΣΤΟΠΟΙΗΣΗ: Εγινε preprocessing στα δεδομένα δημιουργώντας εναν 		  preprocessor τον `class Preprocessor(BaseEstimator, TransformerMixin)``.
             Έγινε αφαίρεση λέξεων χρησιμοποιώντας το regular expression `"\b\w+\b"`
	  Έγινε steming  των λέξεων χρησιμοποιώντας τον PorterStemmer του nltk πακέτου.

	* Για τους αλγορίθμους εκτός των Naive-Bayes  χρησιμοποιήθηκαν τα παρακάτω εργαλεία:

			pipeline = Pipeline([
        				('vec', CountVectorizer(max_features=4096, stop_words='english')),
       				('transformer', TfidfTransformer()),
        				('svd', TruncatedSVD(n_components=40)),
        				('clf', algorithm)
    			])

	  Τα 40 components επιλέχθηκαν μετα απο πειραματισμό και παρατήρηση των scores των classification αλγορίθμων.
	* Για τους Naive-Bayes αλγορίθμους αφαιρέθηκε ο TruncatedSVD διότι παρήγαγε αρνητικές τιμές που επηρρέαζαν αρνητικά τους αλγορίθμους.
	* Εχουμε υλοποιήσει τον αλγόριθμο ΚΝΝ αλλα έχουμε κρατήσει και την υλοποίηση του scikit-learn για λόγουσ σύγκρισης.

```
ΤΑ METRICS precision, recall, f1-score, support υπολογίζονται και τυπώνονται για όλους τους αλγορίθμους κατα την διάρκεια της εκτέλεσης του script classifications.py
```

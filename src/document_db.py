import math
import pickle

import numpy as np
import pandas as pd
import nltk
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from src.config import DATA_PATH
from src.helpers import calculate_cosine_similarity
from src.speed_tester import SpeedTester, OperationType

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')


class DocumentsDB:
    def __init__(self, data_path=DATA_PATH, load_pickled_data=False, use_inverted_index=False):
        if load_pickled_data: # to not calculate tfidf everytime, just unpickle data
            print("Loading data from pickle file...")
            self.load_pickled_data(data_path)
            self.__update_memory_usage()
            self.speed_tester = SpeedTester()
            self.use_inverted_index = use_inverted_index
            print("Data loaded!")
            return

        self.data_path = data_path
        self.use_inverted_index = use_inverted_index
        self.speed_tester = SpeedTester()

        self.full_documents = []  # Stores original document texts (string)

        self.documents_dict = {}  # helper variable for storing internal ids with (wikipedia_id, document_text)

        self.preprocessed_documents = []  # Stores preprocessed documents (tokenized, stemmed, lemmatized lists)
        self.vocabulary = []  # Unique words across all documents
        self.document_word_freq = {}  # {document_id: {word: frequency}}
        self.idf_document_freq = defaultdict(int)  # {word: document frequency (df)}
        self.word_max_f = {}  # {word: max frequency of this word in any document (f)}
        self.tfidf_matrix = None
        self.inverted_index = defaultdict(list)

        # Memory usage
        self.tfidf_matrix_nbytes = None
        self.inverted_index_nbytes = None

        # Initialize the document processing pipeline
        self.speed_tester.start()
        print("Loading documents...")
        self.__load_documents()
        print("Loaded documents...")
        self.speed_tester.stop(OperationType.DATA_LOAD)

        self.speed_tester.start()
        print("Preprocessing documents...")
        self.__preprocess_documents()
        print("Preprocessed documents...")
        self.speed_tester.stop(OperationType.DATA_PREPROCESS)

        self.speed_tester.start()
        self.__calculate_helpers()
        print("Building tfidf matrix...")
        self.__build_tfidf_matrix()
        print("Builded tfidf matrix...")
        self.speed_tester.stop(OperationType.TFIDF_MATRIX_BUILD)

        self.speed_tester.start()
        print("Building inverted index...")
        self.__build_inverted_index()
        print("Builded inverted index...")
        self.speed_tester.stop(OperationType.INVERTED_INDEX_BUILD)

        self.__update_memory_usage()

    def __load_documents(self):
        """Loads documents from CSV file and converts them to a list of strings.

        Additionally store internal document id with wikipedia id and document text.
        """
        documents = pd.read_csv(self.data_path)
        self.full_documents = documents["document_text"].astype(str).tolist()
        for doc_id, (wikipedia_id, text) in enumerate(zip(documents['wikipedia_id'], documents['document_text'])):
            self.documents_dict[doc_id] = (wikipedia_id, str(text))

    def __preprocess_documents(self):
        """Tokenizes, removes stopwords, applies stemming and lemmatization, and updates vocabulary."""
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        for document in self.full_documents:
            document = document.lower()
            tokens = word_tokenize(document)
            tokenized_document = []
            for token in tokens:
                if token.isalpha() and token not in stop_words:
                    lemma_token = lemmatizer.lemmatize(token)  # Lemmatization
                    stemmed_token = stemmer.stem(lemma_token)  # Stemming
                    tokenized_document.append(stemmed_token)
                    self.vocabulary.append(stemmed_token)
            self.preprocessed_documents.append(tokenized_document)

        self.vocabulary = sorted(set(self.vocabulary))

    def __calculate_helpers(self):
        """Computes word frequency, document frequency (df), and max frequency (f) for TF-IDF calculation."""
        for i, document in enumerate(self.preprocessed_documents):
            counter = Counter(document)
            self.document_word_freq[i] = counter  # Store word frequency for the document
            unique_words = counter.keys()

            # Count in how many documents the word appears (df)
            for word in unique_words:
                self.idf_document_freq[word] += 1

            # Track max frequency per word
            for word, freq in counter.items():
                self.word_max_f[word] = max(self.word_max_f.get(word, 0), freq)

    def __calculate_tfidf(self, document_id, word):
        """Calculates the TF-IDF score for a given word in a specific document."""
        # Term Frequency (TF)
        f = self.document_word_freq[document_id][word]
        tf = f / self.word_max_f[word]

        # Inverse Document Frequency (IDF)
        df = self.idf_document_freq[word]
        idf = np.log2(len(self.preprocessed_documents) / df)

        return tf * idf

    def __build_tfidf_matrix(self):
        """Constructs the TF-IDF matrix where rows represent documents and columns represent words in the vocabulary."""
        self.tfidf_matrix = np.zeros((len(self.preprocessed_documents), len(self.vocabulary)))
        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}

        self.document_norms = defaultdict(float)
        for row, document in enumerate(self.preprocessed_documents):
            for word in document:
                col = self.word_to_index[word]
                tf_idf_score = self.__calculate_tfidf(row, word)
                self.tfidf_matrix[row, col] = tf_idf_score

        self.document_norms = {doc_index: np.linalg.norm(doc_vector) for doc_index, doc_vector in
                               enumerate(self.tfidf_matrix)} # precalculate norms

    # TODO
    def __build_inverted_index(self):
        self.inverted_index = defaultdict(list)

        # Create inverted index with (document_id, tfidf score for current document and term)
        for doc_id, document_vector in enumerate(self.tfidf_matrix):
            non_zero_indices = np.nonzero(document_vector)[0]
            for index in non_zero_indices:
                self.inverted_index[index.item()].append((doc_id, self.tfidf_matrix[doc_id, index]))

        # Sort the postings lists by tf-idf score in descending order
        for word in self.inverted_index:
            self.inverted_index[word].sort(key=lambda x: x[0], reverse=False)

    def __update_memory_usage(self):
        """Updates memory usage statistics for TF-IDF matrix and inverted index."""
        self.tfidf_matrix_nbytes = self.tfidf_matrix.nbytes

        inverted_index_size = 0
        for word in self.inverted_index:
            inverted_index_size += len(self.inverted_index[word]) * (
                    np.dtype('int').itemsize + np.dtype('float64').itemsize)

        self.inverted_index_nbytes = inverted_index_size

    # TODO
    def __get_similar_documents_inverted_index(self, document_id, n, print_results=False):
        """Retrieves the top-N most similar documents using inverted index search."""
        query_vector = self.tfidf_matrix[document_id]
        query_norm = self.document_norms[document_id]
        non_zero_indices = np.nonzero(query_vector)[0] # For even faster time, iterate through non zero values only
        scores = defaultdict(float)

        # Iterate through non null terms and get scores for each document with inner product
        for indices in non_zero_indices:
            for doc_id, doc_tfidf in self.inverted_index[indices]:
                if doc_id != document_id:
                    scores[doc_id] += query_vector[indices] * doc_tfidf

        for doc_id in scores:
            scores[doc_id] /= (query_norm * self.document_norms[doc_id])

        sorted_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [doc_id for doc_id, _ in sorted_documents]

    def __get_similar_documents_linear(self, document_id, n, print_results):
        """Retrieves the top-N most similar documents using TF-IDF matrix."""
        documents_id_similarity = []
        for i, document in enumerate(self.tfidf_matrix):
            if i != document_id:
                cosine_similarity = calculate_cosine_similarity(document, self.tfidf_matrix[document_id])
                documents_id_similarity.append((cosine_similarity, i))

        if print_results:
            # Extract cosine similarity scores
            cosine_values = [doc[0] for doc in sorted(documents_id_similarity, reverse=True)[:n]]
            print(cosine_values)

        # Extract document IDs
        most_similar_documents = [doc[1] for doc in sorted(documents_id_similarity, reverse=True)[:n]]
        return most_similar_documents

    def get_document_text(self, document_id):
        """Returns the original text of a document by its ID."""
        return self.full_documents[document_id]

    def get_similar_documents(self, document_id, n=5, print_results=False):
        """Retrieves the top-n most similar documents to the given document ID."""
        self.speed_tester.start()
        if self.use_inverted_index:
            result = self.__get_similar_documents_inverted_index(document_id, n, print_results)
        else:
            result = self.__get_similar_documents_linear(document_id, n, print_results)
        self.speed_tester.stop(OperationType.SEARCH_SIMILAR_DOCUMENTS)
        return result

    def get_memory_usage(self):
        """Return memory usage of different data structures."""
        return self.tfidf_matrix_nbytes, self.inverted_index_nbytes

    def get_speed_statistics(self):
        """Returns object SpeedTester with speed statistics for different data structures and data sizes."""
        return self.speed_tester

    def save_pickled_data(self):
        """Save necessary variables to pickle file. Purely for faster initialization."""
        data = {
            'tfidf_matrix': self.tfidf_matrix,
            'documents_dict': self.documents_dict,
            'vocabulary': self.vocabulary,
            'document_norms': self.document_norms,
            'inverted_index': self.inverted_index
        }
        from config import BASE_DIR
        path = BASE_DIR / f"pickled_data_{len(self.full_documents)}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved pickled data to: {path}")

    def load_pickled_data(self, path):
        """Load pickled data. Purely for faster initialization."""
        data = pickle.load(open(path, 'rb'))
        self.tfidf_matrix = data['tfidf_matrix']
        self.vocabulary = data['vocabulary']
        self.documents_dict = data['documents_dict']
        self.document_norms = data['document_norms']
        self.inverted_index = data['inverted_index']

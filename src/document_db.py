import numpy as np
import pandas as pd
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from src.config import DATA_PATH
from src.helpers import calculate_cosine_similarity

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

class DocumentsDB:
    def __init__(self, data_path=DATA_PATH, use_inverted_index=False):
        self.data_path = data_path
        self.use_inverted_index = use_inverted_index

        self.full_documents = [] # Stores original document texts (string)
        self.preprocessed_documents = [] # Stores preprocessed documents (tokenized, stemmed, lemmatized lists)
        self.vocabulary = [] # Unique words across all documents
        self.document_word_freq = {} # {document_id: {word: frequency}}
        self.idf_document_freq = defaultdict(int) # {word: document frequency (df)}
        self.word_max_f = {} # {word: max frequency of this word in any document (f)}
        self.tfidf_matrix = None
        # TODO self.inverted_index

        # Memory usage
        self.tfidf_matrix_nbytes = None
        self.inverted_index_nbytes = None

        # Initialize the document processing pipeline
        self.load_documents()
        self.preprocess_documents()
        self.calculate_helpers()
        self.build_tfidf_matrix()
        # self.build_inverted_index()
        self.update_memory_usage()

    def load_documents(self):
        """Loads documents from CSV file and converts them to a list of strings."""
        documents = pd.read_csv(self.data_path)
        self.full_documents = documents["document_text"].astype(str).tolist()

    def preprocess_documents(self):
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
                    lemma_token = lemmatizer.lemmatize(token) # Lemmatization
                    stemmed_token = stemmer.stem(lemma_token) # Stemming
                    tokenized_document.append(stemmed_token)
                    self.vocabulary.append(stemmed_token)
            self.preprocessed_documents.append(tokenized_document)

        self.vocabulary = sorted(set(self.vocabulary))

    def calculate_helpers(self):
        """Computes word frequency, document frequency (df), and max frequency (f) for TF-IDF calculation."""
        for i, document in enumerate(self.preprocessed_documents):
            unique_words = set(document)
            word_freq = defaultdict(int)

            for word in document:
                if word in unique_words:
                    self.idf_document_freq[word] += 1 # Count in how many documents the word appears (df)
                    unique_words.remove(word)
                word_freq[word] += 1 # Count word occurrences in the current document

            # Track max frequency per word
            for word, freq in word_freq.items():
                self.word_max_f[word] = max(word_freq.get(word, 0), freq)

            # Store word frequency for the document
            self.document_word_freq[i] = word_freq

    def calculate_tfidf(self, document_id, word):
        """Calculates the TF-IDF score for a given word in a specific document."""
        # Term Frequency (TF)
        f = self.document_word_freq[document_id][word]
        tf = f / self.word_max_f[word]

        # Inverse Document Frequency (IDF)
        df = self.idf_document_freq[word]
        idf = np.log2(len(self.preprocessed_documents) / df)

        return tf * idf

    def build_tfidf_matrix(self):
        """Constructs the TF-IDF matrix where rows represent documents and columns represent words in the vocabulary."""
        self.tfidf_matrix = np.zeros((len(self.preprocessed_documents), len(self.vocabulary)))

        for row, document in enumerate(self.preprocessed_documents):
            for word in document:
                col = self.vocabulary.index(word)
                self.tfidf_matrix[row, col] = self.calculate_tfidf(row, word)

    # TODO
    def build_inverted_index(self):
        ...

    def update_memory_usage(self):
        """Updates memory usage statistics for TF-IDF matrix and inverted index."""
        self.tfidf_matrix_nbytes = self.tfidf_matrix.nbytes
        # TODO self.inverted_index_nbytes

    def get_document_text(self, document_id):
        """Returns the original text of a document by its ID."""
        return self.full_documents[document_id]

    def get_similar_documents(self, document_id, n=5, print_results=False):
        """Retrieves the top-n most similar documents to the given document ID."""
        if self.use_inverted_index:
            return self.get_similar_documents_inverted_index(document_id, n, print_results)
        else:
            return self.get_similar_documents_linear(document_id, n, print_results)

    # TODO
    def get_similar_documents_inverted_index(self, document_id, n, print_results):
        ...

    def get_similar_documents_linear(self, document_id, n, print_results):
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
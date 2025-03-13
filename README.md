# BI-VWM.21


## Possible documents datasets links
https://www.lateral.io/resources-blog/the-unknown-perils-of-mining-wikipedia

Wikipedia query with page id: https://en.wikipedia.org/?curid={page_id}

## Project Structure

DocumentsDatabase
self.documents = numpy.dtype()
self.inverted_index = numpy.dtype() or dict()
self.document_data = dictionary # documents with texts
self.use_inverted_index = True/False

self.documents.nbytes # memory
self.inverted_index.nbytes # memory

deg get_document_text(id):
    return string

def get_similar_documents(query):
    if self.use_inverted_index:
        get_similar_documents_inverted_index
    else
        get_similar_documents_linear

    return list of similar documents ids

def get_similar_documents_linear(query):
    return list of ids of similar documents

def get_similar_documents_inverted_index(query)
    return list of ids of similar documents

SearchEngine
self.current_search_method = [Sequential, Inverted Index]

def display_all_documents():
    return all documents with text

def get_similar_documents(document_id)
    self.current_search_method.get_similar_documents(document_id)
    
    return {
        document1: text
        document2: text,

        time_taken: milliseconds/seconds 
    }

utils.py:
all preprocessing (term, stopwords, stemming etc.)
save to documents folder

app:
server side with / with all documents and /{document_id} that will display chosen document and 

frontend
display document and similar documents

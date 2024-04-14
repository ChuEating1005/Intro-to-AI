import nltk
import string
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer


def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    
    # TO-DO 0: Other preprocessing function attemption
    # Begin your code 
    preprocessed_text = preprocessed_text.lower() # Convert to lower case
    preprocessed_text = preprocessed_text.replace("<br />", " ") # Replacing newline symbol with spaces
    preprocessed_text = preprocessed_text.translate(str.maketrans('', '', string.punctuation)) # Removing punctuation
    stemmer = PorterStemmer()
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(preprocessed_text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    preprocessed_text = ' '.join(stemmed_tokens)
    # End your code

    return preprocessed_text
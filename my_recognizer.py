import warnings
import traceback
import sys

from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    for test_item in range(test_set.num_items):
        #print("test item is : {}".format(test_item))
        probabilities[test_item]=dict()
        for model_word in models:
            X,lengths=test_set.get_item_Xlengths(test_item)
            try:
                logL = models[model_word].score(X,lengths)
            except:
                print
                "Exception in user code:"
                print
                '-' * 60
                traceback.print_exc(file=sys.stdout)
                print
                '-' * 60
                logL=float('-inf')
            probabilities[test_item][model_word]=logL

        max_logL=None
        for word in probabilities[test_item]:
            if max_logL is None:
                max_logL=probabilities[test_item][word]
                guess=word
            if probabilities[test_item][word] > max_logL:
                max_logL=probabilities[test_item][word]
                guess=word
        guesses[test_item]=guess


    return probabilities, guesses

    raise NotImplementedError

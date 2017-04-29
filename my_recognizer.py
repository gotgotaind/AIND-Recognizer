import warnings
import traceback
import sys
import re

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
    transmat_fail=0
    nb_scored=0
    for test_item in range(test_set.num_items):
        #print("test item is : {}".format(test_item))
        probabilities.append(dict())
        for model_word in models:
            X,lengths=test_set.get_item_Xlengths(test_item)
            try:
                nb_scored=nb_scored+1
                logL = models[model_word].score(X,lengths)
            except Exception as inst:
                if re.match("^rows of transmat_ must sum to 1.0",str(inst)):
                    #Ok. That's a known issue...
                    transmat_fail=transmat_fail+1
                    pass
                else:
                    print("Exception {}\nSetting logL to -inf".format(inst))
                # print
                # '-' * 60
                # traceback.print_exc(file=sys.stdout)
                # print
                # '-' * 60
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
        guesses.append(guess)

    print('Got {} transmat failed for a total of {} score calculations'.format(transmat_fail,nb_scored))
    return probabilities, guesses

    raise NotImplementedError

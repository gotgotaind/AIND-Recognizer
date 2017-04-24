import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences



class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        #called by model = SelectorCV(sequences, Xlengths, word,
        #            min_n_components=2, max_n_components=15, random_state = 14).select()
        best_logL=None
        best_model=None





        for num_hidden_states in range(self.min_n_components,self.max_n_components+1):

            logLtotal=0

            # If the number of sequences is < 5, get the number of splits equal to the number
            # of sequences.
            # If more, split 80%/20%
            nb_seq=len(self.sequences)
            if nb_seq<=5:
                nb_folds=nb_seq
            else:
                nb_folds=5
            split_method = KFold(nb_folds)

            #print("Training model for {} with {} hidden states".format(self.this_word, num_hidden_states))

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                #print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx,                                                                          cv_test_idx))  # view indices of the folds
                X_train,lengths_train=combine_sequences(cv_train_idx, self.sequences)
                X_score, lengths_score = combine_sequences(cv_test_idx, self.sequences)

                try:
                    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X_train, lengths_train)
                    logL = model.score(X_score, lengths_score)
                    #print("logL = {}".format(logL))
                except:
                    logL=float("-inf")
                    #print("Training or scoring failed in model for {} with {} hidden states".format(self.this_word, model.n_components))

                logLtotal=logLtotal+logL

            logL=logLtotal
            #print("total logL summed for all kfolds with {} hidden states : {}".format(num_hidden_states,logL))

            if (best_logL is None):
                best_logL=logL
                best_model=model

            if (logL>best_logL):
                best_logL=logL
                best_model = model

        print("Best model found for word {} is with {} hidden states with score {}".format(self.this_word,best_model.n_components,best_logL))

        return best_model

        raise NotImplementedError

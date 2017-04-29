import math
import statistics
import warnings
import re
import traceback
import sys

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
        best_bic=None
        best_model=None

        # N is the number of data points
        N = len(self.X)

        # f is the number of features
        f = len(self.X[0])
        #print("number of features is {}".format(f))

        for num_hidden_states in range(self.min_n_components,self.max_n_components+1):
            try:
                # todo : Would be a good idea to score the model with a part of the data it is
                # not trained with
                model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(self.X,self.lengths)
                logL = model.score(self.X,self.lengths)
            except:
                logL=float("-inf")

            # According to http://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn.hmm.GaussianHMM
            # The parameters of the model are :
            #   startprob_prior : array, shape (n_components, )
            #   transmat_prior : array, shape (n_components, n_components)
            #   means_prior, means_weight : array, shape (n_components, )
            #   covars_prior, covars_weight : array, shape (n_components, )
            # So p the number of parameters is :
            # p=num_hidden_states*3+num_hidden_states*num_hidden_states

            #the free transition probability parameters, which is the size of the transmat matrix less one row because they add up to 1 and therefore the final row is deterministic, so m*(m-1)+ the free starting probabilities, which is the size of startprob minus 1 because it adds to 1.0 and last one can be calculated so m-1 + number of means, which is m*f number of covariances which is the size of the covars matrix, which for "diag" is m*f

            m=num_hidden_states
            p = m*m + 2*m*f - 1



            bic=-2*logL+p*np.log(N)


            if (best_bic is None):
                best_bic=bic
                best_model=model
            # The lower the BIC is the better BIC!
            if (bic<best_bic):
                best_bic=bic
                best_model = model

        print("Best model found for word {} is with {} hidden states with score {}".format(self.this_word,best_model.n_components,best_bic))

        return best_model

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
        best_dic=None
        best_model=None

        # M is the number words in the training set
        M = len(self.hwords.keys())

        for num_hidden_states in range(self.min_n_components,self.max_n_components+1):
            anti_logL = 0
            try:
                # todo : Would be a good idea to score the model with a part of the data it is
                # not trained with
                model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(self.X,self.lengths)
                logL = model.score(self.X,self.lengths)
                for word in self.hwords:
                    if word != self.this_word:
                        X, lengths = self.hwords[word]
                        anti_logL=anti_logL+model.score(X, lengths)
            except:
                logL=float("-inf")

            # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
            #print("logL is {} and antilogL {} and M {}".format(logL,anti_logL,M))
            dic=logL-1.0/(M-1)*anti_logL

            if (best_dic is None):
                best_dic=dic
                best_model=model
            # The higher the DIC is the better DIC!
            if (dic>best_dic):
                best_dic=dic
                best_model = model

        print("Best model found for word {} is with {} hidden states with score {}".format(self.this_word,best_model.n_components,best_dic))

        return best_model
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

        sequences = self.sequences
        nb_seq = len(sequences)

        # If there is only one sequence for this word. We'll duplicate the sequence
        # so that Kfold will work by just doing train set equal score set equal to
        # this single sequence
        if nb_seq == 1:
            sequences = sequences + sequences
            nb_seq = 2
            print("there is only one sequence for {}".format(self.this_word))

        # If the number of sequences is < 5, get the number of splits equal to the number
        # of sequences.
        # If more, split 80%/20%
        if nb_seq <= 5:
            nb_folds = nb_seq
        else:
            nb_folds = 5

        split_method = KFold(nb_folds)

        # Initialize some counters to evaluate the rate of training/scoring failures
        n_samples_fail = 0
        transmat_fail = 0
        nb_scored = 0

        for num_hidden_states in range(self.min_n_components,self.max_n_components+1):

            logLtotal=0

            for cv_train_idx, cv_test_idx in split_method.split(sequences):

                X_train,lengths_train=combine_sequences(cv_train_idx, sequences)
                X_score, lengths_score = combine_sequences(cv_test_idx, sequences)

                try:
                    nb_scored = nb_scored + 1
                    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X_train, lengths_train)
                    logL = model.score(X_score, lengths_score)

                    #print("logL = {}".format(logL))
                except Exception as inst:
                    if re.match("^rows of transmat_ must sum to 1.0", str(inst)):
                        # Ok. That's a known issue...
                        transmat_fail = transmat_fail + 1
                        pass
                    elif re.match("^n_samples",str(inst)):
                        # Ok. That's a known issue...
                        n_samples_fail = n_samples_fail + 1
                        pass

                    else:
                        print("Exception {}\nSetting logL to -inf".format(inst))
                        print("was trying to fit/score with {} hidden states while there were only {} nb_seq".format(num_hidden_states,nb_seq))
                        print
                        '-' * 60
                        traceback.print_exc(file=sys.stdout)
                        print
                        '-' * 60
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
        if transmat_fail>0:
            print('In selectorCV for {}, Got {} transmat failed for a total of {} fit/score calculations'.format(self.this_word,transmat_fail, nb_scored))
        if n_samples_fail>0:
            print('In selectorCV for {}, Got {} n_samples_fail failed for a total of {} fit/score calculations'.format(self.this_word,n_samples_fail, nb_scored))
        print("Best model found for word {} is with {} hidden states with score {}".format(self.this_word,best_model.n_components,best_logL))

        return best_model

        raise NotImplementedError

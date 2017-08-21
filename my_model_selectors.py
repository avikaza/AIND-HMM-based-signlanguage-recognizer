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
        best_score = float("-inf")
        best_model = None
        num_features = self.X.shape[1]

        #L is likelihood of the fitted model, p is the number of parameters, and N is the number of data points
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                new_model = self.base_model(num_states)
                logL = new_model.score(self.X, self.lengths)
                logN = np.log(len(self.X))

                # from discussion forum
                initial_state_occupation_probabilities = num_states
                transition_probabilities = num_states * (num_states - 1)
                emission_probabilities = num_states * num_features * 2
                p = initial_state_occupation_probabilities + transition_probabilities + emission_probabilities

                #calculate the score
                new_score = -2 * logL + p * logN
                if new_score > best_score:
                    best_score = new_score
                    best_model = new_model
            except:
                pass
            
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        best_score = float("-inf")
        best_model = None
        num_features = self.X.shape[1]
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                new_model = self.base_model(num_states)
                logL = new_model.score(self.X, self.lengths)

                #calculate average scor for all other words
                partial_score = 0
                count = 0
                for word in self.words:
                    if word != self.this_word:
                        new_X, new_lengths = self.hwords[word]
                        try:
                            partial_score += hmm_model.score(new_X, new_lengths)
                            count += 1
                        except:
                            pass
                if count > 0:
                    log_all_but_i = partial_score/count
                else:
                    log_all_but_i = 0
            
                #calculate the total score
                new_score = logL - log_all_but_i
                if new_score > best_score:
                    best_score = new_score
                    best_model = new_model
            except:
                pass
        
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection using CV
        n_splits = 3
        if len(self.sequences) < 2:
            return None
        elif len(self.sequences) == 2:
            n_splits = 2
        
        split_method = KFold(n_splits=n_splits)

        best_score = float("-inf")
        best_model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            partial_score = 0
            i = 0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                new_model = None
                try:
                    #Combining sequences with the train split
                    train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                    
                    #Creating the HMM model
                    new_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, random_state=self.random_state).fit(train_X, train_lengths)
                    
                    #Add the test data
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    
                    #Calculate score with the added test data
                    partial_score += hmm_model.score(test_X, test_lengths)
                    i+=1
                except:
                    pass                
            if i > 0:
                new_score = partial_score / i
            else:
                new_score = 0
            if new_score > best_score:
                best_score = new_score
                best_model = new_model
            
        return best_model
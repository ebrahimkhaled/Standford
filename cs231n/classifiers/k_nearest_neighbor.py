import time #add by Ebrahim
t_0 = time.time()
from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange
from scipy import stats   #add by Ebrahim
t0 = time.time()
print ( "time take to import data" , t_0 - t0 )
class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """
    tss1 = time.time()
    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        ts1 = time.time()
        self.X_train = X
        self.y_train = y
        ts2 = time.time() ;print ("train" , ts2-ts1)

    def predict(self, X, k=1, num_loops=0):
        ts1 = time.time()        
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        
        ts2 = time.time() ;print ("predict" , ts2-ts1)
        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        ts1 = time.time()  
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                dists[i,j] = np.sqrt(  np.sum(  (X[i]  -  self.X_train[j] )**2  ) )
                """NOTE:"""
                #el mafrod 2nana bana2s coloum mn coloumn ... bs hna howa mn el bdaia mada5l el mawdo3 bl sha2lob
                #howa mada5 pixil el sora el wa7da fe el row, fkol row bisawie sora ... el sa7 2n kol coloum yab2a baisawi sowa ... bs msh moham
                pass

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ts2 = time.time() ;print ("Twoloops" , ts2-ts1)
        return dists
    tss2 = time.time() 
    print ("_init_" , tss2-tss1)
    def compute_distances_one_loop(self, X):
        ts1 = time.time()  
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        t1 = time.time()
        print("time taken to entre the function" , t1-t0 )

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        t2 = time.time()
        print ( "Time take to make the matrix is :", (t2 - t1) )
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            dists[i,:] = np.sqrt (  np.sum (  np.power(self.X_train - X[i] ,2)   ,1)      )
            pass
        t3 = time.time()
        print ( "Time finish to make the matrix is :", (t3-t2) ) 
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ts2 = time.time() 
        print ("oneloops" , ts2-ts1)
        return dists

    def compute_distances_no_loops(self, X):
        ts1 = time.time()  
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #dists = np.sqrt( np.sum( (X[: ,np.newaxis , : ] - self.X_train)**2  , 1)  )
        dists = (X[: ,np.newaxis , : ] - self.X_train)**2 

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        y_pred_all = np.zeros((num_test,k) )
        closest_y = np.zeros( (num_test,k) )	
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
		
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            '''lw 2ftaradna 2n k=1, el fekra kolaha 2anan 7ansta5d argsort elli bta5od 2a5od el values w taratb el index bta3ha accending mn el so3'air 
            lel kber, fana lo 2a5d 2awel index yab2a ka2anie gabt index 2as3'ar distance, fadl b2a 23rf howa 500 rakm dol el index da ellie mn 
            biatraw7 0 to 4999 tab3 2nhie group (Cat (0) , Dog (1) , Car (2) , ..... ) [w da mmkn na3rfo mn y_train] w hana yege el ahmait el satr el 
            tanie ellie biatkalm 2nana 7na5od el 2arkam de el hia mn 0 to 4999 w na3tberha index el y_train wada5alo arkam 3ala 2naha mask ya5od index w
            yadene el label (Cat 0 , Dog 1 , Car 2 , ....... ) 
            2ama b2a lw aftradna 2n el K NOT = 1, fsa3t-ha 7iab2a el fe kaza 2agaba , kaza label, ana mafroud 7a5od ellie biatkarar fehom '''
            closest_y[i] = np.argsort(dists[i])[0:k] 
            '''w el k=1 yab2a [0:1] yab2a ka2naha [0] ya3ni hatlie 2awel rakm ''' 
            if k != 1 :
                for j in range(k) :
                    y_pred_all[i,j] =  self.y_train[ closest_y[i,j].astype(np.int16) ]
            #print (y_pred_all[i])
            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        if k !=1 : y_pred =  stats.mode(y_pred_all,1)[0]   #3ashan de btrag3 2tnan matrix wa7da bl modes w el tanie 3dd el takrarat ... w 3amalt ,1) 3ashan tagibly el2la fe kol row msh column
        else :  y_pred =  self.y_train[ closest_y.astype(np.int16) ]           
      

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        print (y_pred )
        ts1 = time.time() ;print ("Predict_labels" , ts2-ts1)
        return y_pred

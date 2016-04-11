from kmeans import *
import sys
import matplotlib.pyplot as plt


#plt.ion()

def ShowMeans(means):
    """Show the cluster centers as images."""
    plt.figure(1)
    plt.clf()
    for i in xrange(means.shape[1]):
        plt.subplot(1, means.shape[1], i + 1)
        plt.imshow(means[:, i].reshape(16, 16).T, cmap=plt.cm.gray)
    plt.draw()
    plt.savefig("cluster.png")
    raw_input('Press Enter.')


def mogEM(x, K, iters, minVary=0):
    """
    Fits a Mixture of K Gaussians on x.
    Inputs:
      x: data with one data vector in each column.
      K: Number of Gaussians.
      iters: Number of EM iterations.
      minVary: minimum variance of each Gaussian.

    Returns:
      p : probabilities of clusters.
      mu = mean of the clusters, one in each column.
      vary = variances for the cth cluster, one in each column.
      logProbX = log-probability of data after every iteration.
    """
    N, T = x.shape

    # Initialize the parameters
    randConst = 1.5
    p = randConst + np.random.rand(K, 1)
    p = p / np.sum(p)
    mn = np.mean(x, axis=1).reshape(-1, 1)
    vr = np.var(x, axis=1).reshape(-1, 1)

    # Change the initializaiton with Kmeans here
    # --------------------  Add your code here --------------------
    means = KMeans(x, K, 5)
    mu = means
    # mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)

    # ------------------------------------------------------------
    vary = vr * np.ones((1, K)) * 2
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    logProbX = np.zeros((iters, 1))

    # Do iters iterations of EM
    for i in xrange(iters):
        # Do the E step
        respTot = np.zeros((K, 1))
        respX = np.zeros((N, K))
        respDist = np.zeros((N, K))
        logProb = np.zeros((1, T))
        ivary = 1 / vary
        logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
        logPcAndx = np.zeros((K, T))
        for k in xrange(K):
            dis = (x - mu[:, k].reshape(-1, 1)) ** 2
            logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:, k].reshape(-1, 1) * dis, axis=0)

        mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1)
        mx = np.max(logPcAndx, axis=0).reshape(1, -1)
        PcAndx = np.exp(logPcAndx - mx)
        Px = np.sum(PcAndx, axis=0).reshape(1, -1)
        PcGivenx = PcAndx / Px
        logProb = np.log(Px) + mx
        logProbX[i] = np.sum(logProb)

        print 'Iter %d logProb %.5f' % (i, logProbX[i])

        # Plot log prob of data
        plt.figure(1);
        plt.clf()
        plt.plot(np.arange(i), logProbX[:i], 'r-')
        plt.title('Log-probability of data versus # iterations of EM')
        plt.xlabel('Iterations of EM')
        plt.ylabel('log P(D)');
        plt.draw()

        respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
        respX = np.zeros((N, K))
        respDist = np.zeros((N, K))
        for k in xrange(K):
            respX[:, k] = np.mean(x * PcGivenx[k, :].reshape(1, -1), axis=1)
            respDist[:, k] = np.mean((x - mu[:, k].reshape(-1, 1)) ** 2 * PcGivenx[k, :].reshape(1, -1), axis=1)

        # Do the M step
        p = respTot
        mu = respX / respTot.T
        vary = respDist / respTot.T
        vary = (vary >= minVary) * vary + (vary < minVary) * minVary

        if i==5:
            ShowMeans(mu)
            ShowMeans(vary)

    return p, mu, vary, logProbX


def mogLogProb(p, mu, vary, x):
    """Computes logprob of each data vector in x under the MoG model specified by p, mu and vary."""
    K = p.shape[0]
    N, T = x.shape
    ivary = 1 / vary
    logProb = np.zeros(T)
    for t in xrange(T):
        # Compute log P(c)p(x|c) and then log p(x)
        logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
                    - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
                    - 0.5 * np.sum(ivary * (x[:, t].reshape(-1, 1) - mu) ** 2, axis=0).reshape(-1, 1)

        mx = np.max(logPcAndx, axis=0)
        logProb[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx;
    return logProb #log prob of each pixel?


def q3():
    iters = 5
    minVary = 0.01
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')

    mogEM(inputs_train, 20, 5, 0.01)

    # Train a MoG model with 20 components on all 600 training
    # vectors, with both original initialization and kmeans initialization.
    # ------------------- Add your code here ---------------------

    raw_input('Press Enter to continue.')


def q4():
    iters = 10
    minVary = 0.01
    errorTrain = np.zeros(4)
    errorTest = np.zeros(4)
    errorValidation = np.zeros(4)
    print(errorTrain)
    numComponents = np.array([2, 5, 15, 25])
    T = numComponents.shape[0]
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
    train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
    train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)


    # splitting data into two sets (2s and 3s)

    inputs_list=[inputs_train, inputs_valid, inputs_test]
    target_list= [target_train, target_valid, target_test]
    listof_errorlists= []

    for i in range(0,len(inputs_list)):
        data_input=inputs_list[i]
        targets= target_list[i]
        N= targets.shape[1]



        class_zero = []
        class_one = []
        for i in range(0, N):
            if targets[:, i] == 0:
                class_zero.append(inputs_train[:, i])
            else:
                class_one.append(inputs_train[:, i])

        # turn into arrays
        class_zero = train2
        class_one = train3

        incorrect_zero=0
        incorrect_one=0
        error_list=[]
        for t in xrange(T):
            K = numComponents[t]
            # Train a MoG model with K components for digit 2
            # -------------------- Add your code here --------------------------------



            p0, mu0, vary0, logProbX0 = mogEM(class_zero, K, iters, 0.01)




            # Train a MoG model with K components for digit 3
         




            p, mu, vary, logProbX = mogEM(class_one, K, iters, 0.01)



            # Caculate the probability P(d=1|x) and P(d=2|x),
            # classify examples, and compute error rate
       

            #mogLogProb(p, mu, vary, class_zero)
            #posterior_zero = np.exp(mogLogProb(p0, mu0, vary0, class_zero))
            moglog0 = np.exp(mogLogProb(p0, mu0, vary0, data_input))
            #posterior_zero=((mogLogProb(p0, mu0, vary0, class_zero))*(0.5))/((mogLogProb(p0, mu0, vary0, class_zero))*0.5)+(1-mogLogProb(p0, mu0, vary0, class_zero)*0.5)




            #mogLogProb(p, mu, vary, class_one)
            #posterior_one = np.exp(mogLogProb(p, mu, vary, class_one))
            moglog1 = np.exp(mogLogProb(p, mu, vary, data_input))
            posterior_one=0.5*moglog1/((moglog1*0.5)+(moglog0*0.5))
            posterior_zero=0.5*moglog0/((moglog1*0.5)+(moglog0*0.5))


            N =data_input.shape[1]
            correct = 0
            total = 0
            for i in range(0, N):
                if posterior_zero[i] >= 0.5 and targets.T[i]==0:
                    correct += 1
                if targets.T[i]==0:
                    total+=1

                    incorrect_zero = (total - correct) / float(total)
            print incorrect_zero

            N = data_input.shape[1]
            correct = 0
            total = 0
            for i in range(0, N):
                if posterior_one[i] >= 0.5 and targets.T[i]==1:
                    correct += 1
                if targets.T[i]==1:
                    total+=1
                    incorrect_one = (total - correct) / float(total)
            print incorrect_one



            average_error=(incorrect_zero + incorrect_one)/2
            error_list.append(average_error)

        print error_list
        listof_errorlists.append(error_list)


    # Plot the error rate

   #making cluster into a list of 3 lists

    listof_numcomponents= []
    for i in range (0,len(inputs_list)):
        listof_numcomponents.append(listof_errorlists)

    plt.clf()
 




    plt.figure(1)
    plt.clf()
    listof_labels= ['Training', 'Valid', 'Test']
    legends=[]
    print " "
    print " "
    for i in range(0,len(listof_errorlists)):
        print listof_labels[i]
        print listof_errorlists[i]
        legend= plt.plot(numComponents, listof_errorlists[i], label=listof_labels[i])
        legends.append(legend)
    #plt.legend(legends,listof_labels)
    plt.legend()
    plt.title('Number of Clusters vs. Average Classification Error Rate')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Average Classification Error Rate');
    plt.draw()
    #plt.plot()
    plt.savefig("MoG.png")
    #print listof_errorlists

    #plt.draw()
    raw_input('Press Enter to continue.')




def q5():
    # Choose the best mixture of Gaussian classifier you have, compare this
    # mixture of Gaussian classifier with the neural network you implemented in
    # the last assignment.

    # Train neural network classifier. The number of hidden units should be
    # equal to the number of mixture components.

    # Show the error rate comparison.


    raw_input('Press Enter to continue.')


if __name__ == '__main__':
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')

    #q3()
    q4()
    # q5()

# split data into 2s and 3s
# play around with randconst
# execute mogem a few times , and it will get diff errors (the one that gives highest log prob)




# print "this is class zero"
# mogEM(class_zero, 2, 10, 0.01)

# print "this is class one"
# mogEM(class_one, 2, 10, 0.01)

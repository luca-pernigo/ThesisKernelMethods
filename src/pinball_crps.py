# function to compute pinball score

from sklearn.metrics import mean_pinball_loss



def pinball(tau, Yt_pred, Yt):
    """
    - quantile tau
    - predicted Yt_pred
    - true Yt
    """
    # use sklearn to compute pinball at a certain quantile tau
    pinball_tau=mean_pinball_loss(Yt, Yt_pred, alpha=tau)
    return pinball_tau


def CRPS(Y_pred_test, Y_test):
    """
    Y_test= observed Y for t in test:=[T, T']
    Y_pred_test=predicted for Y_test
    """
    CRPS=0
    # for every t in test
    for t in range(len(Y_test)):
        # for every tau in [0.01, ..., 0.99]
        for tau in range(1,100):
            # discrete approximation of CPRS
            CRPS+=pinball(tau/100, Y_pred_test[t], Y_test[t])
    
    CRPS/=len(Y_test)

    return CRPS



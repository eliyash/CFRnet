import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing

SQRT_CONST = 1e-10

FLAGS = tf.app.flags.FLAGS

def validation_split(D_exp, val_fraction):
    """ Construct a train/validation split """
    n = D_exp['x'].shape[0]

    if val_fraction > 0:
        n_valid = int(val_fraction*n)
        n_train = n-n_valid
        I = np.random.permutation(range(0,n))
        I_train = I[:n_train]
        I_valid = I[n_train:]
    else:
        I_train = range(n)
        I_valid = []

    return I_train, I_valid

def log(logfile,str):
    """ Log a string in a file """
    with open(logfile,'a') as f:
        f.write(str+'\n')
    print str

def save_config(fname):
    """ Save configuration """
    flagdict = {k: v.value for k,v in FLAGS.__dict__['__wrapped'].__dict__['__flags'].items()}
    s = '\n'.join(['%s: %s' % (k, str(flagdict[k])) for k in sorted(flagdict.keys())])
    f = open(fname, 'w')
    f.write(s)
    f.close()


differential_features = ['drug_1.0', 'drug_2.0', 'drug_3.0', 'drug_4.0', 'drug_5.0', 'drug_7.0', 'drug_8.0']


def get_differential_values_counts(df, differential_features):
    # returns an array with the number of samples from each (one hot encoded) feature from a lost
    return [len(df[df[feature] == 1]) for feature in differential_features]


# the function gets a data frame, a list of one-hot encoded features and  a thresould and returns a filtered data
# frame without values that are rare
def exclude_negligible_differential_features(df, differential_features, minumum_number_of_samples=100):
    counts = get_differential_values_counts(df, differential_features)
    to_keep = list(differential_features)
    for i, c in enumerate(counts):
        if (c < minumum_number_of_samples):
            df = df[df[differential_features[i]] != 1]
            df = df.drop([differential_features[i]], axis=1)
            to_keep.remove(differential_features[i])
    return df, to_keep


def normalize_features_in_df(df, features, cofficient=3):
    x = df[features].values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = (min_max_scaler.fit_transform(
        x) * cofficient).astype(
        int)  # todo generalize, check the effect of the multipaction and esxtract to a different fuction
    df[features] = pd.DataFrame(x_scaled)
    return df


def get_data(path_path):
    df_all = pd.read_csv(path_path).drop(columns={'subjectkey'})
    # normalize the qids total variable (since it has a wide range )
    df_all = normalize_features_in_df(df_all, ["qstot"])
    # only consider differential features that are common
    df_all, to_keep = exclude_negligible_differential_features(df_all, differential_features)
    return df_all, len(df_all), to_keep


def keep_2_drags(all_data_df, drug1, drug2):
    what_to_keep = (all_data_df[drug1] > 0) | (all_data_df[drug2] > 0)
    return all_data_df[what_to_keep], [drug1, drug2]


CHOSEN_BINARY_DRUGS = {'drug_1.0', 'drug_2.0'}


def generate_new_dataset(path_path):
    all_data_df, n_train, t_keys = get_data(path_path)

    all_data_df, t_keys = keep_2_drags(all_data_df, *CHOSEN_BINARY_DRUGS)

    y_key = 'remsn'
    t_as_one_hot = all_data_df[t_keys]
    remsn = all_data_df[[y_key]]
    x = all_data_df.drop(columns=t_keys + [y_key])

    def get_t(row):
        for ind, val in enumerate(row):
            if val == 1:
                return ind

    t = t_as_one_hot.apply(get_t, axis=1)

    y = t_as_one_hot * np.repeat(np.array(remsn), t_as_one_hot.shape[-1], axis=1)
    return np.array(x), np.array(y), np.array(t).astype('int')


def load_data(fname):
    """ Load data set """
    if fname[-3:] == 'csv':
        chosen_index = 0
        x, y, t = generate_new_dataset(fname)
        t_binary = np.zeros(t.shape + (1,))
        t_binary[t != chosen_index, :] = 0
        t_binary[t == chosen_index, :] = 1
        remission_sum = y.sum(1)
        # y_binary = np.zeros((y.shape[0] + (2,)))
        # y_binary[:, 0] = remission_sum - y[:, chosen_index]
        # y_binary[:, 1] = y[:, chosen_index]
        y_binary = np.zeros((y.shape[0], 1))
        y_binary[:, 0] = remission_sum
        data = {'x': x, 't': t_binary, 'yf': y_binary, 'ycf': None}

    elif fname[-3:] == 'npz':
        data_in = np.load(fname)
        data = {'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf']}
        try:
            data['ycf'] = data_in['ycf']
        except:
            data['ycf'] = None
    else:
        if FLAGS.sparse>0:
            data_in = np.loadtxt(open(fname+'.y',"rb"),delimiter=",")
            x = load_sparse(fname+'.x')
        else:
            data_in = np.loadtxt(open(fname,"rb"),delimiter=",")
            x = data_in[:,5:]

        data['x'] = x
        data['t'] = data_in[:,0:1]
        data['yf'] = data_in[:,1:2]
        data['ycf'] = data_in[:,2:3]

    data['HAVE_TRUTH'] = not data['ycf'] is None

    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]

    return data

def load_sparse(fname):
    """ Load sparse data set """
    E = np.loadtxt(open(fname,"rb"),delimiter=",")
    H = E[0,:]
    n = int(H[0])
    d = int(H[1])
    E = E[1:,:]
    S = sparse.coo_matrix((E[:,2],(E[:,0]-1,E[:,1]-1)),shape=(n,d))
    S = S.todense()

    return S

def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))

def lindisc(X,p,t):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    mean_control = tf.reduce_mean(Xc,reduction_indices=0)
    mean_treated = tf.reduce_mean(Xt,reduction_indices=0)

    c = tf.square(2*p-1)*0.25
    f = tf.sign(p-0.5)

    mmd = tf.reduce_sum(tf.square(p*mean_treated - (1-p)*mean_control))
    mmd = f*(p-0.5) + safe_sqrt(c + mmd)

    return mmd

def mmd2_lin(X,t,p):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    mean_control = tf.reduce_mean(Xc,reduction_indices=0)
    mean_treated = tf.reduce_mean(Xt,reduction_indices=0)

    mmd = tf.reduce_sum(tf.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def mmd2_rbf(X,t,p,sig):
    """ Computes the l2-RBF MMD for X given t """

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    Kcc = tf.exp(-pdist2sq(Xc,Xc)/tf.square(sig))
    Kct = tf.exp(-pdist2sq(Xc,Xt)/tf.square(sig))
    Ktt = tf.exp(-pdist2sq(Xt,Xt)/tf.square(sig))

    m = tf.to_float(tf.shape(Xc)[0])
    n = tf.to_float(tf.shape(Xt)[0])

    mmd = tf.square(1.0-p)/(m*(m-1.0))*(tf.reduce_sum(Kcc)-m)
    mmd = mmd + tf.square(p)/(n*(n-1.0))*(tf.reduce_sum(Ktt)-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*tf.reduce_sum(Kct)
    mmd = 4.0*mmd

    return mmd

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X,tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X),1,keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y),1,keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D

def pdist2(X,Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X,Y))

def pop_dist(X,t):
    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    M = pdist2(Xt,Xc)
    return M

def wasserstein(X,t,p,lam=10,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt,Xc))

    # M = tf.Print(M, [tf.shape(Xt), tf.shape(Xc), tf.shape(M), tf.shape(M[0:1,:])], "r, c, m, d - shapes")
    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    M_drop = tf.nn.dropout(M,10/(nc*nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam/M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta*tf.ones(tf.shape(M[0:1,:]))
    col = tf.concat([delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))], 0)
    # row = tf.Print(row, [tf.shape(row), tf.shape(col), tf.shape(M), tf.shape(M[0:1,:])], "r, c, m, d - shapes")

    Mt = tf.concat([M,row], 0)
    Mt = tf.concat([Mt,col], 1)

    ''' Compute marginal vectors '''
    a = tf.concat([p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))/nt, (1-p)*tf.ones((1,1))], 0)
    b = tf.concat([(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))/nc, p*tf.ones((1,1))], 0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
    v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))

    T = u*(tf.transpose(v)*K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T*Mt
    D = 2*tf.reduce_sum(E)

    return D, Mlam

def simplex_project(x,k):
    """ Projects a vector x onto the k-simplex """
    d = x.shape[0]
    mu = np.sort(x,axis=0)[::-1]
    nu = (np.cumsum(mu)-k)/range(1,d+1)
    I = [i for i in range(0,d) if mu[i]>nu[i]]
    theta = nu[I[-1]]
    w = np.maximum(x-theta,0)
    return w

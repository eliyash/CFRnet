import tensorflow as tf
import numpy as np
import sys, os
import getopt
import random
import datetime
import traceback

from sklearn.metrics import roc_auc_score

import cfr.cfr_net as cfr
from cfr.util import *

''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('n_in', 2, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 2, """Number of regression layers. """)
tf.app.flags.DEFINE_float('p_alpha', 1e-4, """Imbalance regularization param. """)
tf.app.flags.DEFINE_float('p_lambda', 0.0, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 1, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('dropout_in', 0.9, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 0.9, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'relu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate', 0.05, """Learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.5, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.app.flags.DEFINE_integer('dim_in', 100, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'none', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
tf.app.flags.DEFINE_integer('experiments', 1, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 2000, """Number of iterations. """)
tf.app.flags.DEFINE_float('weight_init', 0.01, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.95, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('wass_iterations', 20, """Number of iterations in Wasserstein computation. """)
tf.app.flags.DEFINE_float('wass_lambda', 1, """Wasserstein lambda. """)
tf.app.flags.DEFINE_integer('wass_bpt', 0, """Backprop through T matrix? """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_string('outdir', '../results/tfnet_topic/alpha_sweep_22_d100/', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', '../data/topic/csv/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'topic_dmean_seed_%d.csv', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', '', """Test data filename form. """)
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_integer('use_p_correction', 1, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_string('optimizer', 'RMSProp', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_string('imb_fun', 'mmd_lin', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
tf.app.flags.DEFINE_integer('output_csv',0,"""Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', -1, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('debug', 0, """Debug mode. """)
tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
tf.app.flags.DEFINE_float('val_part', 0, """Validation part. """)
tf.app.flags.DEFINE_boolean('split_output', 0, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_boolean('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)

if FLAGS.sparse:
    import scipy.sparse as sparse

NUM_ITERATIONS_PER_DECAY = 100

__DEBUG__ = False
if FLAGS.debug:
    __DEBUG__ = True

CHOSEN_BINARY_DRUGS = {'drug_1.0', 'drug_2.0'}


def train(CFR, sess, train_step, D, I_valid, D_test, logfile, i_exp):
    """ Trains a CFR model on supplied data """

    ''' Train/validation split '''
    n = D['x'].shape[0]
    I = range(n); I_train = list(set(I)-set(I_valid))
    n_train = len(I_train)

    ''' Compute treatment probability'''
    p_treated = np.mean(D['t'][I_train,:])

    ''' Set up loss feed_dicts'''
    dict_factual = {CFR.x: D['x'][I_train,:], CFR.t: D['t'][I_train,:], CFR.y_: D['yf'][I_train,:], \
      CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, \
      CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}

    if FLAGS.val_part > 0:
        dict_valid = {CFR.x: D['x'][I_valid,:], CFR.t: D['t'][I_valid,:], CFR.y_: D['yf'][I_valid,:], \
          CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, \
          CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}

    if D['HAVE_TRUTH']:
        dict_cfactual = {CFR.x: D['x'][I_train,:], CFR.t: 1-D['t'][I_train,:], CFR.y_: D['ycf'][I_train,:], \
          CFR.do_in: 1.0, CFR.do_out: 1.0}

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []

    ''' Compute losses '''
    losses = []
    obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],\
      feed_dict=dict_factual)

    cf_error = np.nan
    if D['HAVE_TRUTH']:
        cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

    valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
    if FLAGS.val_part > 0:
        valid_obj, valid_f_error, valid_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],\
          feed_dict=dict_valid)

    losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])

    objnan = False

    reps = []
    reps_test = []

    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):

        ''' Fetch sample '''
        I = random.sample(range(0, n_train), FLAGS.batch_size)
        x_batch = D['x'][I_train,:][I,:]
        t_batch = D['t'][I_train,:][I]
        y_batch = D['yf'][I_train,:][I]

        if __DEBUG__:
            M = sess.run(cfr.pop_dist(CFR.x, CFR.t), feed_dict={CFR.x: x_batch, CFR.t: t_batch})
            log(logfile, 'Median: %.4g, Mean: %.4f, Max: %.4f' % (np.median(M.tolist()), np.mean(M.tolist()), np.amax(M.tolist())))

        ''' Do one step of gradient descent '''
        if not objnan:
            feed_dict = {CFR.x: x_batch, CFR.t: t_batch,
                        CFR.y_: y_batch, CFR.do_in: FLAGS.dropout_in, CFR.do_out: FLAGS.dropout_out,
                        CFR.r_alpha: FLAGS.p_alpha, CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}
            sess.run(train_step, feed_dict=feed_dict)
            # check_remission(CFR, x_batch, y_batch, t_batch, sess)
            # check_remission2(CFR, feed_dict, sess)

        ''' Project variable selection weights '''
        if FLAGS.varsel:
            wip = simplex_project(sess.run(CFR.weights_in[0]), 1)
            sess.run(CFR.projection, feed_dict={CFR.w_proj: wip})

        ''' Compute loss every N iterations '''
        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss,f_error,imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],
                feed_dict=dict_factual)

            rep = sess.run(CFR.h_rep_norm, feed_dict={CFR.x: D['x'], CFR.do_in: 1.0})
            rep_norm = np.mean(np.sqrt(np.sum(np.square(rep), 1)))

            cf_error = np.nan
            if D['HAVE_TRUTH']:
                cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

            valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
            if FLAGS.val_part > 0:
                valid_obj, valid_f_error, valid_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist], feed_dict=dict_valid)

            losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])
            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tImb: %.2g,\tVal: %.3f,\tValImb: %.2g,\tValObj: %.2f' \
                        % (obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj)

            print("**************************** TRAIN****")
            check_remission(CFR, x_batch, y_batch, t_batch, sess)
            print("**************************** VALID****")
            check_remission_dict(CFR, dict_valid, sess)

            if FLAGS.loss == 'log':
                # check_remission_dict(CFR, dict_valid, sess)
                y_pred = sess.run(
                    CFR.output,
                    feed_dict={CFR.x: x_batch, CFR.t: t_batch, CFR.do_in: 1.0, CFR.do_out: 1.0}
                )
                y_pred_binary = 1.0*(y_pred > 0.5)
                acc = 100*(1 - np.mean(np.abs(y_batch - y_pred_binary)))
                loss_str += ',\tAcc: %.2f%%' % acc

            log(logfile, loss_str)

            if np.isnan(obj_loss):
                log(logfile,'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True

        ''' Compute predictions every M iterations '''
        if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i==FLAGS.iterations-1:

            y_pred_f = sess.run(CFR.output, feed_dict={CFR.x: D['x'], \
                CFR.t: D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            y_pred_cf = sess.run(CFR.output, feed_dict={CFR.x: D['x'], \
                CFR.t: 1-D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf),axis=1))

            if D_test is not None:
                y_pred_f_test = sess.run(CFR.output, feed_dict={CFR.x: D_test['x'], \
                    CFR.t: D_test['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
                y_pred_cf_test = sess.run(CFR.output, feed_dict={CFR.x: D_test['x'], \
                    CFR.t: 1-D_test['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test),axis=1))

            if FLAGS.save_rep and i_exp == 1:
                reps_i = sess.run([CFR.h_rep], feed_dict={CFR.x: D['x'], \
                    CFR.do_in: 1.0, CFR.do_out: 0.0})
                reps.append(reps_i)

                if D_test is not None:
                    reps_test_i = sess.run([CFR.h_rep], feed_dict={CFR.x: D_test['x'], \
                        CFR.do_in: 1.0, CFR.do_out: 0.0})
                    reps_test.append(reps_test_i)
    if FLAGS.loss == 'l2':
        y_pred = sess.run(
            CFR.output,
            feed_dict={CFR.x: x_batch, CFR.t: t_batch, CFR.do_in: 1.0, CFR.do_out: 1.0}
        )
        y_pred_binary = 1.0*(y_pred > 0.5)
        acc = 100*(1 - np.mean(np.abs(y_batch - y_pred_binary)))
        log(logfile, ',\tlast Acc: %.2f%%' % acc)

    return losses, preds_train, preds_test, reps, reps_test


def calculate_recommended_remission_rate(y_true_all, pred_all, treatments_all):
    best_treatments = []
    remission_recommended_drug = 0
    total_recommended_drug = 0
    print("****************************AUC****")
    y_pred_all = np.array([y_pred[int(t)] for y_pred, t in zip(pred_all, treatments_all)])
    y_true_all = y_true_all.squeeze(1)
    print(roc_auc_score(y_true_all, y_pred_all))
    for y_true, y_pred, treatment in zip(y_true_all, pred_all, treatments_all):
        top_index = np.argmax(y_pred)
        # if for the patient the recommended drug is the actual drug he received
        if top_index == treatment[0]:
            total_recommended_drug += 1
            remission_recommended_drug += y_true
        best_treatments.append(np.array2string(top_index))

    # print the  "best" treatment for each patient
    best_treatments_count = [(treatment, best_treatments.count(treatment)) for treatment in set(best_treatments)]
    for treat, count in best_treatments_count:
        print('treatment {}: {}'.format(treat, count))

    real_remission = y_true_all.mean()
    new_remission = float(remission_recommended_drug) / float(total_recommended_drug)
    print("****************************REMISSION BEFORE RECOMMENDED DRUG********")
    print(real_remission)
    print("****************************REMISSION RATE OF PATIENTS WHO RECEIVED RECOMMENDED DRUG********")
    # print the average of the max prediction
    print(new_remission)
    print(total_recommended_drug)
    print("****************************REMISSION NEW/OLD********")
    print(new_remission/real_remission)


def check_remission_dict(CFR, dict_of_values, sess):
    x = dict_of_values[CFR.x]
    y = dict_of_values[CFR.y_]
    t = dict_of_values[CFR.t]
    check_remission(CFR, x, y, t, sess)


def check_remission2(CFR, feed_dict, sess):
    # print("****************************T1****")
    t = feed_dict[CFR.t]
    feed_dict[CFR.t] = np.zeros_like(t)
    y_pred_t1 = sess.run(
        CFR.output,
        feed_dict=feed_dict
    )
    # print("****************************T2****")
    feed_dict[CFR.t] = np.ones_like(t)
    y_pred_t2 = sess.run(
        CFR.output,
        feed_dict=feed_dict
    )
    # print("****************************END****")
    y_pred_vec = np.concatenate([y_pred_t1, y_pred_t2], axis=1)
    calculate_recommended_remission_rate(feed_dict[CFR.y_], y_pred_vec, t)


def check_remission(CFR, x, y, t, sess):
    # print("****************************T1****")
    y_pred_t1 = sess.run(
        CFR.output,
        feed_dict={CFR.x: x, CFR.t: np.zeros_like(t), CFR.do_in: 1.0, CFR.do_out: 1.0}
    )
    # print("****************************T2****")
    y_pred_t2 = sess.run(
        CFR.output,
        feed_dict={CFR.x: x, CFR.t: np.ones_like(t), CFR.do_in: 1.0, CFR.do_out: 1.0}
    )
    # print("****************************END****")
    y_pred_vec = np.concatenate([y_pred_t1, y_pred_t2], axis=1)
    calculate_recommended_remission_rate(y, y_pred_vec, t)


def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    npzfile = outdir+'result'
    npzfile_test = outdir+'result.test'
    repfile = outdir+'reps'
    repfile_test = outdir+'reps.test'
    outform = outdir+'y_pred'
    outform_test = outdir+'y_pred.test'
    lossform = outdir+'loss'
    logfile = outdir+'log.txt'
    f = open(logfile,'w')
    f.close()
    dataform = FLAGS.datadir + FLAGS.dataform

    has_test = False
    if not FLAGS.data_test == '': # if test set supplied
        has_test = True
        dataform_test = FLAGS.datadir + FLAGS.data_test

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir+'config.txt')

    log(logfile, 'Training with hyperparameters: alpha=%.2g, lambda=%.2g' % (FLAGS.p_alpha,FLAGS.p_lambda))

    ''' Load Data '''
    npz_input = False
    if dataform[-3:] == 'npz':
        npz_input = True
    if npz_input:
        datapath = dataform
        if has_test:
            datapath_test = dataform_test
    else:
        datapath = dataform
        has_test = False
        # datapath = dataform % 1
        # if has_test:
        #     datapath_test = dataform_test % 1

    log(logfile,     'Training data: ' + datapath)
    if has_test:
        log(logfile, 'Test data:     ' + datapath_test)
    D = load_data(datapath)
    D_test = None
    if has_test:
        D_test = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [%d,%d]' % (D['n'], D['dim']))

    ''' Start Session '''
    sess = tf.Session()

    ''' Initialize input placeholders '''
    x  = tf.placeholder("float", shape=[None, D['dim']], name='x') # Features
    t  = tf.placeholder("float", shape=[None, 1], name='t')   # Treatent
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome

    ''' Parameter placeholders '''
    r_alpha = tf.placeholder("float", name='r_alpha')
    r_lambda = tf.placeholder("float", name='r_lambda')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [D['dim'], FLAGS.dim_in, FLAGS.dim_out]
    CFR = cfr.cfr_net(x, t, y_, p, FLAGS, r_alpha, r_lambda, do_in, do_out, dims)

    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(FLAGS.lrate, global_step, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    opt = None
    if FLAGS.optimizer == 'Adagrad':
        opt = tf.train.AdagradOptimizer(lr)
    elif FLAGS.optimizer == 'GradientDescent':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif FLAGS.optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(lr)
    else:
        opt = tf.train.RMSPropOptimizer(lr, FLAGS.decay)

    ''' Unused gradient clipping '''
    #gvs = opt.compute_gradients(CFR.tot_loss)
    #capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
    #train_step = opt.apply_gradients(capped_gvs, global_step=global_step)

    train_step = opt.minimize(CFR.tot_loss, global_step=global_step)

    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []
    if FLAGS.varsel:
        all_weights = None
        all_beta = None

    all_preds_test = []

    ''' Handle repetitions '''
    n_experiments = FLAGS.experiments
    if FLAGS.repetitions>1:
        if FLAGS.experiments>1:
            log(logfile, 'ERROR: Use of both repetitions and multiple experiments is currently not supported.')
            sys.exit(1)
        n_experiments = FLAGS.repetitions

    ''' Run for all repeated experiments '''
    for i_exp in range(1, n_experiments+1):

        if FLAGS.repetitions>1:
            log(logfile, 'Training on repeated initialization %d/%d...' % (i_exp, FLAGS.repetitions))
        else:
            log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))

        ''' Load Data (if multiple repetitions, reuse first set)'''

        if i_exp==1 or FLAGS.experiments>1:
            D_exp_test = None
            if npz_input:
                D_exp = {}
                D_exp['x']  = D['x'][:,:,i_exp-1]
                D_exp['t']  = D['t'][:,i_exp-1:i_exp]
                D_exp['yf'] = D['yf'][:,i_exp-1:i_exp]
                if D['HAVE_TRUTH']:
                    D_exp['ycf'] = D['ycf'][:,i_exp-1:i_exp]
                else:
                    D_exp['ycf'] = None

                if has_test:
                    D_exp_test = {}
                    D_exp_test['x']  = D_test['x'][:,:,i_exp-1]
                    D_exp_test['t']  = D_test['t'][:,i_exp-1:i_exp]
                    D_exp_test['yf'] = D_test['yf'][:,i_exp-1:i_exp]
                    if D_test['HAVE_TRUTH']:
                        D_exp_test['ycf'] = D_test['ycf'][:,i_exp-1:i_exp]
                    else:
                        D_exp_test['ycf'] = None
            else:
                datapath = dataform
                D_exp = load_data(datapath)
                if has_test:
                    datapath_test = dataform_test % i_exp
                    D_exp_test = load_data(datapath_test)

            D_exp['HAVE_TRUTH'] = D['HAVE_TRUTH']
            if has_test:
                D_exp_test['HAVE_TRUTH'] = D_test['HAVE_TRUTH']

        ''' Split into training and validation sets '''
        I_train, I_valid = validation_split(D_exp, FLAGS.val_part)

        ''' Run training loop '''
        losses, preds_train, preds_test, reps, reps_test = \
            train(CFR, sess, train_step, D_exp, I_valid, \
                D_exp_test, logfile, i_exp)

        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_losses.append(losses)

        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train,1,3),0,2)
        if  has_test:
            out_preds_test = np.swapaxes(np.swapaxes(all_preds_test,1,3),0,2)
        out_losses = np.swapaxes(np.swapaxes(all_losses,0,2),0,1)

        ''' Store predictions '''
        log(logfile, 'Saving result to %s...\n' % outdir)
        if FLAGS.output_csv:
            np.savetxt('%s_%d.csv' % (outform,i_exp), preds_train[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (outform_test,i_exp), preds_test[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (lossform,i_exp), losses, delimiter=',')

        ''' Compute weights if doing variable selection '''
        if FLAGS.varsel:
            if i_exp == 1:
                all_weights = sess.run(CFR.weights_in[0])
                all_beta = sess.run(CFR.weights_pred)
            else:
                all_weights = np.dstack((all_weights, sess.run(CFR.weights_in[0])))
                all_beta = np.dstack((all_beta, sess.run(CFR.weights_pred)))

        ''' Save results and predictions '''
        all_valid.append(I_valid)
        if FLAGS.varsel:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, w=all_weights, beta=all_beta, val=np.array(all_valid))
        else:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        if has_test:
            np.savez(npzfile_test, pred=out_preds_test)

        ''' Save representations '''
        if FLAGS.save_rep and i_exp == 1:
            np.savez(repfile, rep=reps)

            if has_test:
                np.savez(repfile_test, rep=reps_test)


def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir+'/results_'+timestamp+'/'
    os.mkdir(outdir)

    try:
        run(outdir)
    except Exception as e:
        with open(outdir+'error.txt','w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == '__main__':
    argv = \
        'cfr_net_train.py' \
        ' --weight_init 0.1 --varsel 0 --dim_in 20 --dim_out 10 --nonlin elu --use_p_correction 0 ' \
        '--dropout_out 0.5 --rbf_sigma 0.1 --dropout_in 0.5 --n_in 5 ' \
        '--datadir ./data/ --val_part 0.2 ' \
        '--lrate_decay 0.97 --reweight_sample 1 --optimizer Adam --imb_fun wass --split_output 1 --repetitions 1 ' \
        '--wass_iterations 10 ' \
        '--pred_output_delay 200 --normalization divide --outdir ./results/example_ihdp ' \
        '--rep_weight_decay 0 --wass_bpt 1 --decay 0.3 --sparse 0 --n_out 3 ' \
        '--batch_norm 0 ' \
        '--experiments 1 ' \
        '--batch_size 200 ' \
        '--iterations 100000 ' \
        '--wass_lambda 10.0 ' \
        '--p_lambda 0.000 ' \
        '--lrate 0.001 ' \
        '--p_alpha 0.001 ' \
        '--loss l2 ' \
        '--dataform transformed_all_data.csv'
        # '--p_alpha 1 ' \
        # ' --p_lambda 0.0001 ' \
        # '--iterations 100000 ' \
        # '--dataform ihdp_npci_1-100.train.npz'
    tf.app.run(argv=argv.split(' '))
    # tf.app.run()

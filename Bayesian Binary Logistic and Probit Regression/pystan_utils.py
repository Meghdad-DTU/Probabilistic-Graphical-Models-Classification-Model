import numpy as np
from matplotlib import pyplot as plt
from pystan.external.pymc import plots
import seaborn as sns
import sys

if sys.version_info[0] == 3:
    def xrange(i):
        return range(i)

def vb_extract(fit):
    var_names = fit["sampler_param_names"][:-1]
    samples = np.array([x for x in fit["sampler_params"]])
    
    samples_dict = {}
    means_dict = {}
    for i in xrange(len(var_names)):
        samples_dict[var_names[i]] = samples[i,:]
        means_dict[var_names[i]] = fit["mean_pars"][i]
        
    return samples_dict, means_dict, var_names


def vb_extract_variable(fit, var_name, var_type="real", dims=None):
    if var_type == "real":
        return fit["mean_pars"][fit["sampler_param_names"].index(var_name)]
    elif var_type == "vector":
        vec = []
        for i in xrange(len(fit["sampler_param_names"])):
            if var_name in fit["sampler_param_names"][i]:
                vec.append(fit["mean_pars"][i])
        return np.array(vec)
    elif var_type == "matrix":
        if dims == None:
            raise Exception("For matrix variables, you must specify a 'dims' parameter")
        C, D = dims
        mat = []
        for i in xrange(len(fit["sampler_param_names"])):
            if var_name in fit["sampler_param_names"][i]:
                mat.append(fit["mean_pars"][i])
        mat = np.array(mat).reshape(C, D, order='F')
        return mat
    else:
        raise Exception("Unknown variable type: %s. Valid types are: real, vector and matrix" % (var_type,))


def vb_plot_variables(fit, var_names):
    samples, means, names = vb_extract(fit)

    if type(var_names) == str:
        var_names = [var_names]
    elif type(var_names) != list:
        raise Exception("Invalid argument type for var_names")

    to_plot = []
    for var in var_names:
        for i in xrange(len(fit["sampler_param_names"])):
            if var == fit["sampler_param_names"][i] or var in fit["sampler_param_names"][i]: 
                to_plot.append(fit["sampler_param_names"][i])

    for var in to_plot:
        plots.kdeplot_op(plt, samples[var])
    plt.legend(to_plot)
    plt.show()


def report(fit, prefix=''):
    for param in fit['sampler_param_names']:
        if param.startswith(prefix):
            print(param, "=", vb_extract_variable(fit, var_name=param))
            
def plot_trace(param, param_name='parameter'):
  """Plot the trace and posterior of a parameter."""
  
  # Summary statistics
  mean = np.mean(param)
  median = np.median(param)
  cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)
  
  # Plotting
  plt.subplot(1,2,1)
  plt.plot(param,color="b")
  plt.xlabel('samples')
  plt.ylabel(param_name)
  plt.axhline(mean, color='r', lw=2, linestyle='--')
  plt.axhline(median, color='c', lw=2, linestyle='--')
  plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
  plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
  plt.title('Trace and Posterior Distribution for {}'.format(param_name))

  plt.subplot(1,2,2)
  plt.hist(param, 30, density=True, color="blue",); sns.kdeplot(param, shade=True,color="g")
  plt.xlabel(param_name)
  plt.ylabel('density')
  plt.axvline(mean, color='r', lw=2, linestyle='--',label='mean')
  plt.axvline(median, color='c', lw=2, linestyle='--',label='median')
  plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
  plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)
  
  plt.gcf().tight_layout()
  plt.legend()
    
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
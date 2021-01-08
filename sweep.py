from random import sample

from clearml import Task

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="SurF'N", task_name='Hyper-Parameter Sweep', task_type=Task.TaskTypes.optimizer)

# Create a hyper-parameter dictionary for the task
params = dict()

# track my parameters dictionary
params = task.connect(params)



# define random search space,
params['resample'] = [True, False]
params['repeat'] = [True, False]
params['reiterate'] = [True, False]
params['adv_run_rate'] = [None, 0.1, 0.01, 0.001]
params['selection_dec_rate'] = [None, .999, .99, .9]
params['selection_rate'] = [0.01, 0.05, 0.1]
params['min_adv'] = [0, None]
params['gradient_agg'] = [False, True]
params['do_surfn'] = [True, False]
params['uniform_sampling'] = [False, True]
params['deterministic'] = [False, True]
params['credit_method'] = ["signs", "signs-bt", "signs-gc", "bt-gc", None]
params['credit_method_2'] = ["relu", "abs", "shift-dim", "shift", None]
params['fitness_dist'] = ["sm", "None", "credit", "sum"]
params['probas_dist'] = ["sm", "sq", "sum"]
params['temp'] = [.05, .1, .2, .7]
params['nonzero'] = [True, False]
params["select_method"] = ["numpy", "torch"]


# This is a simple random search
# (can be integrated with 'bayesian-optimization' 'hpbandster' etc.)
space = {
    key: lambda: sample(params[key], 1)[0] for key in params
}

# number of random samples to test from 'space'
params['total_number_of_experiments'] = 1000

# execution queue to add experiments to
params['execution_queue_name'] = 'default'

# experiment template to optimize with random parameter search
params['experiment_template_name'] = "run"

# Select base template task
# Notice we can be more imaginative and use task_id which will eliminate the need to use project name
template_task = Task.get_task(project_name="SurF'N", task_name=params['experiment_template_name'])

for i in range(params['total_number_of_experiments']):
    # clone the template task into a new write enabled task (where we can change parameters)
    cloned_task = Task.clone(source_task=template_task,
                             name=template_task.name+' {}'.format(i), parent=template_task.id)

    # get the original template parameters
    cloned_task_parameters = cloned_task.get_parameters()

    # override with random samples form grid
    for k in space.keys():
        cloned_task_parameters[k] = space[k]()

    # put back into the new cloned task
    cloned_task.set_parameters(cloned_task_parameters)
    print('Experiment {} set with parameters {}'.format(i, cloned_task_parameters))

    # enqueue the task for execution
    Task.enqueue(cloned_task.id, queue_name=params['execution_queue_name'])
    print('Experiment id={} enqueue for execution'.format(cloned_task.id))

# we are done, the next step is to watch the experiments graphs
print('Done')
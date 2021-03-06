# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05b_evaluate.ipynb (unless otherwise specified).

__all__ = ['pl_logger', 'load_model', 'load_model_jupyter', 'proximity', 'cf_gen_parallel', 'model_cf_gen', 'evaluate',
           'test_evaluate']

# Cell
from .import_essentials import *
from .utils import *
from .train import *
from .training_module import *
from .net import *
from .baseline import *

from pytorch_lightning.metrics.functional.classification import *
# imports from captum library
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

plt.style.use(['science', 'ieee'])
pl_logger = logging.getLogger('lightning')

# Cell
def load_model(checkpoint_path: str, n_iter: int, module=BaselineModel, t_configs={'gpus': 0}):
    model = module.load_from_checkpoint(checkpoint_path)
    trainer = pl.Trainer(
        max_epochs=n_iter, resume_from_checkpoint=checkpoint_path, num_sanity_val_steps=0, **t_configs)
    trainer.fit(model)
    return model


def load_model_jupyter(checkpoint_path: str, n_iter: int, module=BaselineModel, m_configs={}, t_configs={'gpus': 0}):
    model = module.load_from_checkpoint(checkpoint_path, configs=m_configs)
    trainer = pl.Trainer(
        max_epochs=n_iter, resume_from_checkpoint=checkpoint_path, num_sanity_val_steps=0, **t_configs)
    trainer.fit(model)
    return model


def proximity(x, c):
    return torch.abs(x - c).sum(dim=1).mean()

# Cell
def cf_gen_parallel(cf_params: dict, CFExplainer: ExplainerBase, is_parallel: bool = True, test_size: int = None) -> Dict:
    """generate CF in parallel

    Args:
        model (BaselineModel): black-box model to be explained
        CFExplainer (ExplainerBase): cf algo to explain the model

    Returns:
        Dict: results
    """
    model = cf_params['model']

    val_dataset = model.val_dataset
    _, label = val_dataset[:]
    length = len(val_dataset) if test_size is None else test_size
    # length = 1
    X = torch.rand((length, val_dataset[0][0].size(-1)))

    cf_algo = torch.rand((length, val_dataset[0][0].size(-1)))
    print(f"y-axis: {val_dataset[0][0].size(-1)}")

    def gen_step(ix, x, y):
        x = x.reshape(1, -1)
        cf = CFExplainer(x, **cf_params)
        # generate counterfactual explanation for algo and model
        _cf_algo = cf.generate_cf(1000)
        return x, _cf_algo.detach()

    result = []
    # run generate cf in parallel
    start = time.time()
    if is_parallel:
        result = Parallel(n_jobs=-1, max_nbytes=None, verbose=False)(
            delayed(gen_step)(
                ix=ix,
                x=x,
                y=y
            )
            for ix, (x, y) in enumerate(tqdm(val_dataset)) if ix < length
        )
    else:
        for ix, (x, y) in enumerate(tqdm(val_dataset)):
            if ix < length:
                x, _cf_algo = gen_step(ix, x, y)
                result.append((x, _cf_algo))

    total_time = time.time() - start
    average_time = total_time / length

    for ix, (x, _cf_algo) in enumerate(result):
        X[ix, :] = x
        cf_algo[ix, :] = _cf_algo

    # validity metrics
    y_prime = torch.ones((length)) - model.predict(X)
    cf_y_algo = model.predict(cf_algo)
    # cf_y_model = model.predict(cf_model)

    # robustness
    diffs, total_num = model.check_cont_robustness(X, cf_algo, cf_y_algo)
    algo_robustness = 1 - torch.true_divide(diffs, total_num) if total_num != 0 else 0.

    # diffs, total_num = model.check_cont_robustness(X, cf_model, cf_y_model)
    # model_robustness = 1 - torch.true_divide(diffs, total_num) if total_num != 0 else 0.
    return {
        "x": X,
        "cf": cf_algo,
        "y_prime": y_prime,
        "cf_y": cf_y_algo,
        "diffs": diffs,
        "total_num": total_num,
        "robustness": algo_robustness,
        "total_time": total_time,
        "average_time": average_time,
        "pred_accuracy": accuracy(label[:length], model.predict(X))
    }


def model_cf_gen(model: CounterfactualTrainingModule, check_speed: bool = False) -> Dict:
    val_dataset = model.val_dataset
    X, y = val_dataset[:]

    start = time.time()
    cf = model.generate_cf(X)
    total_time = time.time() - start

    average_time = -1
    if check_speed:
        start = time.time()
        for x, _ in val_dataset:
            x = x.reshape(1, -1)
            model.generate_cf(x)
        average_time = (time.time() - start) / len(val_dataset)

    # validity metrics
    y_prime = torch.ones(y.size()) - model.predict(X)
    cf_y = model.predict(cf)
    # robustness
    diffs, total_num = model.check_cont_robustness(X, cf, cf_y)
    algo_robustness = 1 - torch.true_divide(diffs, total_num) if total_num != 0 else 0.

    return {
        "x": X,
        "cf": cf,
        "y_prime": y_prime,
        "cf_y": cf_y,
        "diffs": diffs,
        "total_num": total_num,
        "robustness": algo_robustness,
        "total_time": total_time,
        "average_time": average_time,
        "pred_accuracy": accuracy(y, model.predict(X))
    }

# Cell
def evaluate(result: Dict, dataset_name: str, cf_name: str, is_logging: bool = True):
    """calculate metrics of CF algos and log the results

    Args:
        result (Dict): results generated from `cf_gen_parallel`
            - x: input instance
            - cf: counterfactual examples
            - y_prime: desired label (the filp of predicted label when the problem is binary)
            - cf_y: counterfactual outcomes
        dataset_name (str): dataset name
        cf_name (str): counterfactual algorithm's name

    Raises:
        ValueError: dataset name is invalid
        ValueError: cf_name is invalid

    Returns:
        Dict: final result
    """
    x = result['x']
    cf = result['cf']
    y_prime = result['y_prime']
    cf_y = result['cf_y']
    cat_idx = result['cat_idx']
    diffs = result['diffs']
    total_num = result['total_num']
    robustness = result['robustness']
    total_time = result['total_time']
    pred_accuracy = result['pred_accuracy']

    if torch.is_tensor(diffs):
        diffs = diffs.item()
    if torch.is_tensor(total_num):
        total_num = total_num.item()
    if torch.is_tensor(robustness):
        robustness = robustness.item()
    if torch.is_tensor(pred_accuracy):
        pred_accuracy = pred_accuracy.item()

    dataset_names = ['dummy', 'adult', 'student', 'home']
    extra_dataset_names = ['credit', 'german', 'student_performance', 'breast', 'heart', 'titanic']
    cf_names = ['VanillaCF', 'DiverseCF',
                'ProtoCF', 'VAE-CF', 'CounterfactualNet']
    metrics = ['cat_proximity', 'cont_proximity', 'validity',
               'robustness', 'sparsity', 'diffs', 'total_num', 'time', 'pred_accuracy', 'proximity']

    is_extra = dataset_name in extra_dataset_names
    if is_extra:
        csv_path = f"results/extra/{dataset_name}/metrics.csv"
    else:
        csv_path = f"results/{dataset_name}/metrics.csv"

    if is_extra:
        result_path = f"results/extra/{dataset_name}/{cf_name}_result.pt"
    else:
        result_path = f"results/{dataset_name}/{cf_name}_result.pt"


    if (dataset_name not in dataset_names) and (dataset_name not in extra_dataset_names):
        raise ValueError(
            f"dataset_name ({dataset_name}) is not valid; it should be one of {dataset_names + extra_dataset_names}.")

    if cf_name not in cf_names:
        raise ValueError(
            f"cf_name ({cf_name}) is not valid; it should be one of {cf_names}.")

    if os.path.exists(csv_path):
        r = pd.read_csv(csv_path, index_col=0).to_dict()
        for metric in metrics:
            if metric not in r.keys():
                r[metric] = {cf_algo: -1 for cf_algo in cf_names}
    else:
        r = {metric: {cf_algo: -1 for cf_algo in cf_names}
             for metric in metrics}

    r['cont_proximity'][cf_name] = proximity(x[:, :cat_idx], cf[:, :cat_idx]).item()
    r['cat_proximity'][cf_name] = proximity(x[:, cat_idx:], cf[:, cat_idx:]).item()
    r['validity'][cf_name] = accuracy(y_prime, cf_y).item()
    r['robustness'][cf_name] = robustness
    r['diffs'][cf_name] = diffs
    r['total_num'][cf_name] = total_num
    r['time'][cf_name] = total_time
    r['pred_accuracy'][cf_name] = pred_accuracy
    r['proximity'][cf_name] = proximity(x, cf).item()
    # r['sparsity'] = r['cat_proximity'][cf_name] / 2 + r['cont_proximity'][cf_name] - total_num

    if is_logging:
        pd.DataFrame.from_dict(r).to_csv(csv_path)
        torch.save(result, result_path)
        print("metrics have been saved")

    final_result = {metric: r[metric][cf_name] for metric in metrics}

    print("Final result:")
    pprint(final_result)

    return final_result

def test_evaluate():
    result = {
        "x": torch.rand((1000, 127)),
        "cf": torch.rand((1000, 127)),
        "y_prime": torch.rand((1000, 1)),
        "cf_y": torch.rand((1000, 1)),
        "diffs": 100,
        "total_num": 100,
        "robustness": 1.0
    }
    result["cat_idx"] = 21
    evaluate(result, dataset_name="student", cf_name="VanillaCF")
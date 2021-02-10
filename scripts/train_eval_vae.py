from counterfactual.import_essentials import *
from counterfactual.utils import *
from counterfactual.train import *
from counterfactual.training_module import *
from counterfactual.net import *
from counterfactual.evaluate import *
from counterfactual.baseline import *

from torch.nn.parameter import Parameter
from pytorch_lightning.metrics.functional.classification import *


if __name__ == "__main__":
    dummy_config = json.load(open("counterfactual/configs/dummy.json"))
    adult_config = json.load(open("counterfactual/configs/adult.json"))
    student_config = json.load(open("counterfactual/configs/student.json"))
    home_config = json.load(open("counterfactual/configs/home.json"))

    t_config = json.load(open("counterfactual/configs/trainer.json"))

    adult_config['batch_size'] = 2048
    student_config['batch_size'] = 2048
    home_config['batch_size'] = 2048

    t_config['max_epochs'] = 50
    
    ###################################################################
    # adult
    ###################################################################
    model = load_model('saved_weights/adult/baseline/epoch=55-step=10695.ckpt', 56)
    adult_config['validity_reg'] = 0.2
    cf_result = train(
        VAE_CF(adult_config, model=model),
        t_config,
        logger=pl_loggers.TestTubeLogger(Path('log/'), name="adult/vae")
    )

    result = model_cf_gen(cf_result['module'])
    result["cat_idx"] = len(model.continous_cols)
    evaluate(result, dataset_name="adult", cf_name="VAE-CF")

    ###################################################################
    # home
    ###################################################################
    model = load_model('saved_weights/home/baseline/epoch=92-step=5765.ckpt', 93)
    home_config['validity_reg'] = 0.2
    cf_result = train(
        VAE_CF(home_config, model=model),
        t_config,
        logger=pl_loggers.TestTubeLogger(Path('log/'), name="home/vae")
    )

    result = model_cf_gen(cf_result['module'])
    result["cat_idx"] = len(model.continous_cols)
    evaluate(result, dataset_name="home", cf_name="VAE-CF")

    ###################################################################
    # student
    ###################################################################
    model = load_model('saved_weights/student/baseline/epoch=98-step=18908.ckpt', 99)
    student_config['validity_reg'] = 0.2
    cf_result = train(
        VAE_CF(student_config, model=model),
        t_config,
        logger=pl_loggers.TestTubeLogger(Path('log/'), name="student/vae")
    )

    result = model_cf_gen(cf_result['module'])
    result["cat_idx"] = len(model.continous_cols)
    evaluate(result, dataset_name="student", cf_name="VAE-CF")

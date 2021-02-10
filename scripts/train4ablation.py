from counterfactual.import_essentials import *
from counterfactual.utils import *
from counterfactual.train import *
from counterfactual.training_module import *
from counterfactual.net import *
from counterfactual.evaluate import *
from counterfactual.baseline import *


if __name__ == "__main__":

    configs = [
        # {'name': 'adult', 'path': 'counterfactual/configs/adult.json'}
        {'name': 'home', 'path': 'counterfactual/configs/home.json'},
        {'name': 'student', 'path': 'counterfactual/configs/student.json'},
    ]
    # adult_config = json.load(open("counterfactual/configs/adult.json"))
    t_config = json.load(open("counterfactual/configs/trainer.json"))
    student_config = json.load(open("counterfactual/configs/student.json"))

    for config in configs:
        m_config = json.load(open(config['path']))
        train(
            module=CounterfactualModel(m_config),
            t_configs=t_config,
            logger=pl_loggers.TestTubeLogger(Path('log/'), name=f"{config['name']}/ablation"),
            description="single BP"
        )

        m_config.update({
            "loss_1": "cross_entropy",
            "loss_2": "mse",
            "loss_3": "cross_entropy"
        })
        train(
            module=CounterfactualModel2Optimizers(m_config),
            t_configs=t_config,
            logger=pl_loggers.TestTubeLogger(Path('log/'),  name=f"{config['name']}/ablation"),
            description="cross entropy"
        )

        m_config = json.load(open(config['path']))
        m_config['smooth_y'] = False
        train(
            module=CounterfactualModel2Optimizers(m_config),
            t_configs=t_config,
            logger=pl_loggers.TestTubeLogger(Path('log/'),  name=f"{config['name']}/ablation"),
            description="no smoothing"
        )
    # train(
    #     module=CounterfactualModel2Optimizers(student_config),
    #     t_configs=t_config,
    #     logger=pl_loggers.TestTubeLogger(Path('log/'),  name=f"student/ablation"),
    #     description="original"
    # )
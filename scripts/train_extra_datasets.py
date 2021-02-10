from counterfactual.import_essentials import *
from counterfactual.utils import *
from counterfactual.train import *
from counterfactual.training_module import *
from counterfactual.net import *
from counterfactual.evaluate import *
from counterfactual.baseline import *


if __name__ == "__main__":
    breast_config = load_json("counterfactual/configs/extra/breast_cancer.json")
    credit_card_config = load_json("counterfactual/configs/extra/credit_card.json")
    german_credit_config = load_json("counterfactual/configs/extra/german_credit.json")
    student_config = load_json("counterfactual/configs/extra/student_performance.json")
    heart_config = load_json("counterfactual/configs/extra/heart.json")
    titanic_config = load_json("counterfactual/configs/extra/titanic.json")

    t_config = json.load(open("counterfactual/configs/extra/trainer.json"))
    lrs = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

    start_time = time.time()
    ###################################################################
    # breast
    ###################################################################
    baseline = train(
        module=BaselineModel(breast_config),
        t_configs=t_config,
        logger=pl_loggers.TestTubeLogger(Path('log/'), name="extra/breast/baseline")
    )


    for lr in lrs:
        breast_config['lr'] = lr

        train(
            module=CounterfactualModel2Optimizers(breast_config),
            t_configs=t_config,
            logger=pl_loggers.TestTubeLogger(Path('log/'), name="extra/breast/cf_2opt")
        )

    ###################################################################
    # credit
    ###################################################################
    baseline = train(
        module=BaselineModel(credit_card_config),
        t_configs=t_config,
        logger=pl_loggers.TestTubeLogger(Path('log/'), name="extra/credit/baseline")
    )

    for lr in lrs:
        credit_card_config['lr'] = lr

        train(
            module=CounterfactualModel2Optimizers(credit_card_config),
            t_configs=t_config,
            logger=pl_loggers.TestTubeLogger(Path('log/'), name="extra/credit/cf_2opt")
        )

    ###################################################################
    # german
    ###################################################################
    baseline = train(
        module=BaselineModel(german_credit_config),
        t_configs=t_config,
        logger=pl_loggers.TestTubeLogger(Path('log/'), name="extra/german/baseline")
    )

    for lr in lrs:
        german_credit_config['lr'] = lr

        train(
            module=CounterfactualModel2Optimizers(german_credit_config),
            t_configs=t_config,
            logger=pl_loggers.TestTubeLogger(Path('log/'), name="extra/german/cf_2opt")
        )

    ###################################################################
    # student
    ###################################################################
    baseline = train(
        module=BaselineModel(student_config),
        t_configs=t_config,
        logger=pl_loggers.TestTubeLogger(Path('log/'), name="extra/student/baseline")
    )

    for lr in lrs:
        student_config['lr'] = lr

        train(
            module=CounterfactualModel2Optimizers(student_config),
            t_configs=t_config,
            logger=pl_loggers.TestTubeLogger(Path('log/'), name="extra/student/cf_2opt")
        )

    ###################################################################
    # heart
    ###################################################################
    baseline = train(
        module=BaselineModel(heart_config),
        t_configs=t_config,
        logger=pl_loggers.TestTubeLogger(Path('log/'), name="extra/heart/baseline")
    )

    for lr in lrs:
        heart_config['lr'] = lr
        train(
            module=CounterfactualModel2Optimizers(heart_config),
            t_configs=t_config,
            logger=pl_loggers.TestTubeLogger(Path('log/'), name="extra/heart/cf_2opt")
        )

    ###################################################################
    # titanic
    ###################################################################

    baseline = train(
        module=BaselineModel(titanic_config),
        t_configs=t_config,
        logger=pl_loggers.TestTubeLogger(Path('log/'), name="extra/titanic/baseline")
    )

    for lr in lrs:
        titanic_config['lr'] = lr
        train(
            module=CounterfactualModel2Optimizers(titanic_config),
            t_configs=t_config,
            logger=pl_loggers.TestTubeLogger(Path('log/'), name="extra/titanic/cf_2opt")
        )

    print(f"total time: {time.time() - start_time}")
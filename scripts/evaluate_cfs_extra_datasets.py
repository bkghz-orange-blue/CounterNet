from counterfactual.import_essentials import *
from counterfactual.utils import *
from counterfactual.train import *
from counterfactual.training_module import *
from counterfactual.net import *
from counterfactual.evaluate import *
from counterfactual.baseline import *


if __name__ == "__main__" and not in_jupyter():
    breast_config = load_json("counterfactual/configs/extra/breast_cancer.json")
    credit_card_config = load_json("counterfactual/configs/extra/credit_card.json")
    german_credit_config = load_json("counterfactual/configs/extra/german_credit.json")
    student_config = load_json("counterfactual/configs/extra/student_performance.json")
    heart_config = load_json("counterfactual/configs/extra/heart.json")
    titanic_config = load_json("counterfactual/configs/extra/titanic.json")
    dummy_config = load_json("counterfactual/configs/dummy.json")

    t_config = json.load(open("counterfactual/configs/extra/trainer.json"))

    configs = [
        {
            "data_name": "breast", 
            "c_net_path": "saved_weights/extra/breast/c_net/epoch=371-step=1487.ckpt", 
            "c_net_epoch": 372,
            "baseline_path": "saved_weights/extra/breast/baseline/epoch=383-step=1535.ckpt",
            "baseline_epoch": 384,
            "config": breast_config
        },
        {
            "data_name": "credit", 
            "c_net_path": "saved_weights/extra/credit/c_net/epoch=90-step=2001.ckpt", 
            "c_net_epoch": 91,
            "baseline_path": "saved_weights/extra/credit/baseline/epoch=361-step=7963.ckpt",
            "baseline_epoch": 362,
            "config": credit_card_config
        },
        {
            "data_name": "german", 
            "c_net_path": "saved_weights/extra/german/c_net/epoch=21-step=131.ckpt", 
            "c_net_epoch": 22,
            "baseline_path": "saved_weights/extra/german/baseline/epoch=19-step=119.ckpt",
            "baseline_epoch": 20,
            "config": german_credit_config
        },
        {
            "data_name": "heart", 
            "c_net_path": "saved_weights/extra/heart/c_net/epoch=144-step=289.ckpt", 
            "c_net_epoch": 145,
            "baseline_path": "saved_weights/extra/heart/baseline/epoch=78-step=157.ckpt",
            "baseline_epoch": 79,
            "config": heart_config
        },
        {
            "data_name": "student_performance", 
            "c_net_path": "saved_weights/extra/student_performance/c_net/epoch=142-step=571.ckpt", 
            "c_net_epoch": 143,
            "baseline_path": "saved_weights/extra/student_performance/baseline/epoch=287-step=1151.ckpt",
            "baseline_epoch": 288,
            "config": student_config
        },
        {
            "data_name": "titanic", 
            "c_net_path": "saved_weights/extra/titanic/c_net/epoch=26-step=161.ckpt", 
            "c_net_epoch": 27,
            "baseline_path": "saved_weights/extra/titanic/baseline/epoch=63-step=383.ckpt",
            "baseline_epoch": 64,
            "config": titanic_config
        },
        {
            "data_name": "dummy", 
            "c_net_path": "saved_weights/dummy/c_net/epoch=91-step=5427.ckpt", 
            "c_net_epoch": 92,
            "baseline_path": "saved_weights/dummy/baseline/epoch=77-step=4601.ckpt",
            "baseline_epoch": 78,
            "config": dummy_config
        },
    ]

    current_time = time.time()

    for config in configs:
        print("dealing ", config['data_name'])
        model = load_model(config['c_net_path'], config['c_net_epoch'], module=CounterfactualModel2Optimizers)
        result = model_cf_gen(model)
        result["cat_idx"] = len(model.continous_cols)
        evaluate(result, dataset_name=config['data_name'], cf_name="CounterfactualNet")

        # load baseline model
        model = load_model(config['baseline_path'], config['baseline_epoch'])
        result = cf_gen_parallel(CFExplainer=VanillaCF, cf_params={'model': model})
        result["cat_idx"] = len(model.continous_cols)
        evaluate(result, dataset_name=config['data_name'], cf_name="VanillaCF")

        result = cf_gen_parallel(CFExplainer=DiverseCF, cf_params={'model': model})
        result["cat_idx"] = len(model.continous_cols)
        evaluate(result, dataset_name=config['data_name'], cf_name="DiverseCF")

        # train AE first
        t_config['max_epochs'] = 10
        t_config['gpus'] = 0
        result = train(AE(config['config']), t_config)
        ae = result['module']

        result = cf_gen_parallel(CFExplainer=ProtoCF, cf_params={
            'model': model, 'ae': ae, 'train_loader': DataLoader(model.train_dataset, batch_size=128, shuffle=True)
        })
        result["cat_idx"] = len(model.continous_cols)
        evaluate(result, dataset_name=config['data_name'], cf_name="ProtoCF")

        # train VAE 
        config['config']['validity_reg'] = 0.2
        t_config['max_epochs'] = 50
        cf_result = train(
            VAE_CF(config['config'], model=model),
            t_config,
            logger=pl_loggers.TestTubeLogger(Path('log/'), name=f"extra/{config['data_name']}/vae")
        )

        result = model_cf_gen(cf_result['module'])
        result["cat_idx"] = len(model.continous_cols)
        evaluate(result, dataset_name=config['data_name'], cf_name="VAE-CF")

    print("total time: ", time.time() - current_time)
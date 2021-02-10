from counterfactual.import_essentials import *
from counterfactual.utils import *
from counterfactual.train import *
from counterfactual.training_module import *
from counterfactual.net import *
from counterfactual.evaluate import *
from counterfactual.baseline import *


if __name__ == "__main__" and not in_jupyter():
    dummy_config = json.load(open("counterfactual/configs/dummy.json"))
    adult_config = json.load(open("counterfactual/configs/adult.json"))
    student_config = json.load(open("counterfactual/configs/student.json"))
    home_config = json.load(open("counterfactual/configs/home.json"))

    breast_config = load_json("counterfactual/configs/extra/breast_cancer.json")
    credit_card_config = load_json("counterfactual/configs/extra/credit_card.json")
    german_credit_config = load_json("counterfactual/configs/extra/german_credit.json")
    student_performance_config = load_json("counterfactual/configs/extra/student_performance.json")
    heart_config = load_json("counterfactual/configs/extra/heart.json")
    titanic_config = load_json("counterfactual/configs/extra/titanic.json")


    t_config = json.load(open("counterfactual/configs/trainer.json"))
    cf_names = ['VanillaCF', 'DiverseCF',
                'ProtoCF', 'VAE-CF', 'CounterfactualNet']
    csv_path = "results/speed.csv"

    if os.path.exists(csv_path):
        r = pd.read_csv(csv_path, index_col=0).to_dict(orient='index')
    else:
        r = {cf: {} for cf in cf_names}

    configs = [
    #     {
    #         "name": "adult",
    #         "baseline_path":
    #         "saved_weights/adult/baseline/epoch=55-step=10695.ckpt",
    #         "baseline_iter": 56,
    #         "c_net_path": "saved_weights/adult/c_net/epoch=243-step=46603.ckpt",
    #         "c_net_iter": 244
    #     },
    #     {
    #         "name": "home",
    #         "baseline_path":
    #         "saved_weights/home/baseline/epoch=92-step=5765.ckpt",
    #         "baseline_iter": 93,
    #         "c_net_path": "saved_weights/home/c_net/epoch=993-step=61627.ckpt",
    #         "c_net_iter": 994
    #     },
    #     {
    #         "name": "student",
    #         "baseline_path":
    #         "saved_weights/student/baseline/epoch=98-step=18908.ckpt",
    #         "baseline_iter": 99,
    #         "c_net_path": "saved_weights/student/c_net/epoch=767-step=146687.ckpt",
    #         "c_net_iter": 768
    #     },
        {
            "name": "extra/student_performance",
            "baseline_path":
            "saved_weights/extra/student_performance/baseline/epoch=287-step=1151.ckpt",
            "baseline_iter": 288,
            "c_net_path": "saved_weights/extra/student_performance/c_net/epoch=142-step=571.ckpt",
            "c_net_iter": 143,
            "m_config": student_performance_config
        },
        {
            "name": "extra/titanic",
            "baseline_path":
            "saved_weights/extra/titanic/baseline/epoch=63-step=383.ckpt",
            "baseline_iter": 64,
            "c_net_path": "saved_weights/extra/titanic/c_net/epoch=26-step=161.ckpt",
            "c_net_iter": 27,
            "m_config": titanic_config
        },
        {
            "name": "extra/breast",
            "baseline_path":
            "saved_weights/extra/breast/baseline/epoch=383-step=1535.ckpt",
            "baseline_iter": 384,
            "c_net_path": "saved_weights/extra/breast/c_net/epoch=371-step=1487.ckpt",
            "c_net_iter": 372,
            "m_config": breast_config
        },
        
    ]   

    for config in configs:

        model = load_model(config['baseline_path'], config['baseline_iter'])
        # VanillaCF
        result = cf_gen_parallel(CFExplainer=VanillaCF, cf_params={'model': model}, is_parallel=False, test_size = 50)
        r['VanillaCF'][config['name']] = result['average_time']

        # DiverseCF
        result = cf_gen_parallel(CFExplainer=DiverseCF, cf_params={'model': model}, is_parallel=False, test_size = 50)
        r['DiverseCF'][config['name']] = result['average_time']

        # ProtoCF
        # train AE first
        t_config['max_epochs'] = 1
        result = train(AE(config['m_config']), t_config)
        ae = result['module']

        result = cf_gen_parallel(CFExplainer=ProtoCF, cf_params={
            'model': model, 'ae': ae, 'train_loader': DataLoader(model.train_dataset, batch_size=128, shuffle=True)
        }, is_parallel=False, test_size = 50)
        r['ProtoCF'][config['name']] = result['average_time']
        
        # VAE
        config['m_config']['validity_reg'] = 0.2
        config['m_config']['batch_size'] = 1024
        cf_result = train(
            VAE_CF(config['m_config'], model=model),
            t_config,
            logger=pl_loggers.TestTubeLogger(Path('log/'), name=f"{config['name']}/vae")
        )
        result = model_cf_gen(cf_result['module'], check_speed=True)
        r['VAE-CF'][config['name']] = result['average_time']

        # CounterNet
        model = load_model(config['c_net_path'], config['c_net_iter'], module=CounterfactualModel2Optimizers)
        result = model_cf_gen(model, check_speed=True)
        r['CounterfactualNet'][config['name']] = result['average_time']

    print(r)
    pd.DataFrame.from_dict(r, orient="index").to_csv(csv_path)

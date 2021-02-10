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

    t_config = json.load(open("counterfactual/configs/trainer.json"))

    ###################################################################
    # adult
    ###################################################################
    # model = load_model("saved_weights/adult/c_net/epoch=243-step=46603.ckpt", 244, module=CounterfactualModel2Optimizers)
    # result = model_cf_gen(model)
    # result["cat_idx"] = len(model.continous_cols)
    # evaluate(result, dataset_name="adult", cf_name="CounterfactualNet")

    model = load_model("saved_weights/adult/baseline/epoch=55-step=10695.ckpt", 56)
    # result = cf_gen_parallel(CFExplainer=VanillaCF, cf_params={'model': model})
    # result["cat_idx"] = len(model.continous_cols)
    # evaluate(result, dataset_name="adult", cf_name="VanillaCF")

    result = cf_gen_parallel(CFExplainer=DiverseCF, cf_params={'model': model})
    result["cat_idx"] = len(model.continous_cols)
    evaluate(result, dataset_name="adult", cf_name="DiverseCF")

    # train AE first
    t_config['max_epochs'] = 10
    result = train(AE(adult_config), t_config)
    ae = result['module']

    result = cf_gen_parallel(CFExplainer=ProtoCF, cf_params={
        'model': model, 'ae': ae, 'train_loader': DataLoader(model.train_dataset, batch_size=128, shuffle=True)
    })
    result["cat_idx"] = len(model.continous_cols)
    evaluate(result, dataset_name="adult", cf_name="ProtoCF")

    # vae_cf = load_model("saved_weights/adult/vae-cf/epoch=28-step=347.ckpt", 29, module=VAE_CF)
    # result = model_cf_gen(model)
    # result["cat_idx"] = len(model.continous_cols)
    # evaluate(result, dataset_name="adult", cf_name="VAE_CF")

    ###################################################################
    # student
    ###################################################################
    # model = load_model("saved_weights/student/c_net/epoch=767-step=146687.ckpt", 768, module=CounterfactualModel2Optimizers)
    # result = model_cf_gen(model)
    # result["cat_idx"] = len(model.continous_cols)
    # evaluate(result, dataset_name="student", cf_name="CounterfactualNet")

    model = load_model("saved_weights/student/baseline/epoch=98-step=18908.ckpt", 99)
    # result = cf_gen_parallel(CFExplainer=VanillaCF, cf_params={'model': model})
    # result["cat_idx"] = len(model.continous_cols)
    # evaluate(result, dataset_name="student", cf_name="VanillaCF")

    result = cf_gen_parallel(CFExplainer=DiverseCF, cf_params={'model': model})
    result["cat_idx"] = len(model.continous_cols)
    evaluate(result, dataset_name="student", cf_name="DiverseCF")

    # AE
    t_config['max_epochs'] = 10
    result = train(AE(student_config), t_config)
    ae = result['module']

    result = cf_gen_parallel(CFExplainer=ProtoCF, cf_params={
        'model': model, 'ae': ae, 'train_loader': DataLoader(model.train_dataset, batch_size=128, shuffle=True)
    })
    result["cat_idx"] = len(model.continous_cols)
    evaluate(result, dataset_name="student", cf_name="ProtoCF")

    ###################################################################
    # home
    ###################################################################
    # model = load_model("saved_weights/home/c_net/epoch=993-step=61627.ckpt", 994, module=CounterfactualModel2Optimizers)
    # result = model_cf_gen(model)
    # result["cat_idx"] = len(model.continous_cols)
    # evaluate(result, dataset_name="home", cf_name="CounterfactualNet")

    model = load_model("saved_weights/home/baseline/epoch=92-step=5765.ckpt", 93)
    # result = cf_gen_parallel(CFExplainer=VanillaCF, cf_params={'model': model})
    # result["cat_idx"] = len(model.continous_cols)
    # evaluate(result, dataset_name="home", cf_name="VanillaCF")

    result = cf_gen_parallel(CFExplainer=DiverseCF, cf_params={'model': model})
    result["cat_idx"] = len(model.continous_cols)
    evaluate(result, dataset_name="home", cf_name="DiverseCF")

    # AE
    t_config['max_epochs'] = 10
    result = train(AE(home_config), t_config)
    ae = result['module']

    result = cf_gen_parallel(CFExplainer=ProtoCF, cf_params={
        'model': model, 'ae': ae, 'train_loader': DataLoader(model.train_dataset, batch_size=128, shuffle=True)
    })
    result["cat_idx"] = len(model.continous_cols)
    evaluate(result, dataset_name="home", cf_name="ProtoCF")

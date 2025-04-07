import os
import torch
import torch.optim as optim
import itertools
from models import LAMAR
from utils import load_data, set_seed, prepare_dataloader, train_step, eval_step, eval_verite, early_stop

def run_experiment(data_path, 
                   use_multiclass, 
                   recon_list,
                   data_name_v="_llava_v1_3",                    
                   model_version=None, 
                   transformer_version=None, 
                   choose_fusion_method= [["concat_1", "add", "sub", "mul"]],  
                   filter_data_options=["filter_len_0","filter_len_5","filter_len_10", "filter_len_15", "filter_len_25", "filter_len_50",""],
                   pooling_method=False, 
                   sim_coding=False,
                   data_name="newsclippings", 
                   choose_gpu = 0,
                   classifier_version = "mlp_1",
                   num_workers=8,
                   epochs = 50,
                   early_stop_epochs = 10,
                   encoder = "CLIP",
                   encoder_version = "ViT-L/14",
                   activation="gelu",
                   dropout=0.1,
                   batch_size_options = [512],
                   init_model_name = 'LAMAR_',
                   seed_options = [0],        
                   emb_dim=768,
                   lr_options = [1e-4],
                   tf_dim_options = [1024],
                  ):
    
    encoder_version = encoder_version.replace('-', '').replace('/', '') 

    if data_name=="newsclippings" and use_multiclass in [False, True]:
        pass
    else:
        filter_data_options = [""]
    
    for filter_data in filter_data_options:
    
        device = torch.device("cuda:" + str(choose_gpu) if torch.cuda.is_available() else "cpu")
        print(device)
    
        train_data, valid_data, test_data, verite_test, image_embeddings, text_embeddings, verite_image_embeddings, verite_text_embeddings = load_data(data_path, data_name, data_name_v, encoder, encoder_version, filter_data, use_multiclass)   
            
        for use_reconstruction in recon_list:
                
            if use_multiclass == False:
                results_filename = "results_miscaptions_miscaptioned"
                
            elif use_multiclass == "OOC":            
                results_filename = "results_miscaptions_OOC"
                
            elif use_multiclass == True:            
                results_filename = "results_miscaptions_multiclass"        
                        
            chosen_criterion = "bce_clf:1"
            
            if use_reconstruction:
                chosen_criterion = "mse+" + chosen_criterion
                        
            if sim_coding:
                
                sims_to_keep = ['img_txt'] 
    
                if use_reconstruction == False and model_version == None:
                    choose_fusion_method = [[False]]       
                    
            else:
                sims_to_keep = [] 
            
    
            reconstruction_layers = [1] if use_reconstruction else [False]
            sim_coding_layers = [1] if sim_coding else [False]
                    
            if transformer_version == "default":
                tf_heads_layers = [[4, 4, 4, 4]] 
                
            elif transformer_version == "aitr":
                tf_heads_layers = [[1, 2, 4, 8],[8, 4, 2, 1]] 
                
            else:
                tf_heads_layers = [None] 
                tf_dim_options = [None]
    
            grid = itertools.product(choose_fusion_method, reconstruction_layers, sim_coding_layers, batch_size_options, lr_options, tf_heads_layers, tf_dim_options, seed_options)        
            
            experiment = 0
            for params in grid:
            
                fusion_method, num_reconstruction_layers, num_sim_coding_layers, batch_size, learning_rate, tf_h_l, tf_dim, seed = params
            
                set_seed(seed)
            
                valid_data_list = []
                final_verite_list = []        
                
                train_dataloader = prepare_dataloader(input_data=train_data,
                                                      visual_features=image_embeddings,
                                                      textual_features=text_embeddings,
                                                      batch_size=batch_size,
                                                      num_workers=num_workers,
                                                      shuffle=True)        
                
                valid_dataloader = prepare_dataloader(input_data=valid_data,
                                                      visual_features=image_embeddings,
                                                      textual_features=text_embeddings,
                                                      batch_size=batch_size,
                                                      num_workers=num_workers,
                                                      shuffle=True)    
                
                test_dataloader = prepare_dataloader(input_data=test_data,
                                                     visual_features=image_embeddings,
                                                     textual_features=text_embeddings,
                                                     batch_size=batch_size,
                                                     num_workers=num_workers)
                
                verite_dataloader = prepare_dataloader(input_data=verite_test,
                                                       visual_features=verite_image_embeddings,
                                                       textual_features=verite_text_embeddings,
                                                       batch_size=batch_size,
                                                       num_workers=num_workers)    
            
                history = []
                has_not_improved_for = 0
            
                full_model_name = init_model_name + "_" + encoder_version + data_name_v + "_" + str(learning_rate) + "_" + str(emb_dim) + "_" + chosen_criterion + "_" + str(batch_size) + "_" + str(seed) + "_" + data_name + data_name_v + "_" + str(sim_coding) + "_" + str(use_reconstruction) + "_" + str(model_version) + "_" + str(transformer_version) + "_" + str(pooling_method)
            
                parameters = {
                                "LEARNING_RATE": learning_rate,
                                "EPOCHS": epochs, 
                                "BATCH_SIZE": batch_size,
                                "MODEL_VERSION": str(model_version),
                                "TRANSFORMER_VERSION": str(transformer_version),
                                "POOLING_MECHANISM": str(pooling_method), 
                                "TF_H_L": tf_h_l,
                                "TF_DIM": tf_dim,
                                "SIM_CODING" : sim_coding, 
                                "SIM_CODING_LAYERS": num_sim_coding_layers,
                                "SIMS_TO_KEEP": sims_to_keep, 
                                "CLASSIFIER_VERSION": classifier_version,
                                "USE_RECONSTRUCTION": use_reconstruction,
                                "RECONSTRUCTION_LAYERS": reconstruction_layers, 
                                "FUSION_METHOD": fusion_method,
                                "ACTIVATION": activation,
                                "NUM_WORKERS": num_workers,
                                "EARLY_STOP_EPOCHS": early_stop_epochs,
                                "ENCODER": encoder,
                                "ENCODER_VERSION": encoder_version,
                                "SEED": seed,
                                "CRITERION": chosen_criterion, 
                                "FILTERED_DATA": filter_data,
                                "USE_MULTICLASS": use_multiclass,
                                'data_name': data_name,
                                'data_name_v': data_name_v,
                                "full_model_name": full_model_name,
                            }
                
                PATH = "checkpoints_pt/model_" + full_model_name + ".pt"  
                
                model = LAMAR(emb_dim, 
                             fusion_method, 
                             sim_coding, 
                             num_sim_coding_layers, 
                             sims_to_keep, 
                             use_reconstruction, 
                             num_reconstruction_layers, 
                             model_version, 
                             transformer_version,
                             tf_h_l,
                             tf_dim,
                             pooling_method, 
                             classifier_version,
                             use_multiclass,
                             dropout)
            
                model.to(device)
                
                print(model)
                
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                                
                for current_epoch in range(epochs):
                    
                    print("Path:", PATH)
                    
                    train_step(model, 
                               train_dataloader, 
                               use_multiclass,
                               current_epoch, 
                               optimizer,
                               chosen_criterion, 
                               device, 
                               batches_per_epoch=train_dataloader.__len__())
                    
                    if "schedule" in chosen_criterion:
                        scheduler.step()    
                    
                    results = eval_step(model, 
                                        valid_dataloader, 
                                        use_multiclass, 
                                        -1, 
                                        chosen_criterion, 
                                        device, 
                                        batches_per_epoch=valid_dataloader.__len__())      
                                
                    history.append(results)
                            
                    has_not_improved_for = early_stop(
                        has_not_improved_for,
                        model,
                        optimizer,
                        history,
                        current_epoch,
                        PATH,
                        metrics_list=["Accuracy"],
                    )         
                        
                    if has_not_improved_for >= early_stop_epochs:
                
                        print(
                            f"Performance has not improved for {early_stop_epochs} epochs. Stop training at epoch {current_epoch}!"
                        )
                        break
            
                print("Finished Training. Loading the best model from checkpoints.")
                
                checkpoint = torch.load(PATH)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                epoch = checkpoint["epoch"]
                            
                res_val = eval_step(model, 
                                    valid_dataloader, 
                                    use_multiclass,  
                                    -1, 
                                    chosen_criterion, 
                                    device, 
                                    batches_per_epoch=valid_dataloader.__len__())  
            
                res_test = eval_step(model, 
                                     test_dataloader, 
                                     use_multiclass,
                                     -1, 
                                     chosen_criterion, 
                                     device, 
                                     batches_per_epoch=test_dataloader.__len__())
                
                
                verite_results = eval_verite(model, 
                                             verite_test, 
                                             verite_dataloader, 
                                             use_multiclass, 
                                             device
                                            )
                    
                res_verite = {
                    "verite_" + str(key.lower()): val for key, val in verite_results.items()
                }
            
                    
                res_val = {
                    "valid_" + str(key.lower()): val for key, val in res_val.items()
                }
                
                res_test = {
                    "test_" + str(key.lower()): val for key, val in res_test.items()
                }
                
                all_results = {**parameters, **res_test, **res_val, **res_verite}
                
                all_results["path"] = PATH
                all_results["history"] = history
                
                if not os.path.isdir("results"):
                    os.mkdir("results")
            
                save_results_csv(
                    "results/",
                    results_filename,            
                    all_results,
                )        
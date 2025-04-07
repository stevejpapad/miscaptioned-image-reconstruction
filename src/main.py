from utils import extract_features
from experiment import run_experiment

data_path = "/"
images_path="VisualNews/origin/"

# Extract features for "Miscaption This!"
for data_version in ["_llava_v1_1", "_llava_v1_2", "_llava_v1_3", "_llava_v1_4"]:
    extract_features(
        data_path,
        images_path, # path for VisualNews    
        data_name="newsclippings", 
        data_name_v=data_version
    )

# Results for Figure 4
# Performance of detection models (DT-Transformer and RED-DOT) trained on four variations of the "MisCaption This!" dataset (D1, D2, D3, D4) under varying filtering thresholds (l ∈ 0, 5, 10, 15, 25, 50, None) 
for data_version in ["_llava_v1_1", "_llava_v1_2", "_llava_v1_3", "_llava_v1_4"]:
    run_experiment(data_path=data_path, 
                   data_name_v=data_version, 
                   use_multiclass=False, # Task: True vs Miscaptioned
                   recon_list=[False], # No reconstruction
                   model_version="transformer", 
                   transformer_version="default", 
                   choose_fusion_method=[["concat_1", "add", "sub", "mul"], ["concat_1"]], # For RED-DOT and DT-Transformer. respectively 
                   filter_data_options=["filter_len_0","filter_len_5","filter_len_10", "filter_len_15", "filter_len_25", "filter_len_50",""]
                  )


# Experiments for Table 1
# COMPARATIVE AND ABLATION ANALYSIS OF THE PROPOSED LAMAR AND PRIOR SOTA MODELS (DT-TRANSFORMER, RED-DOT, AND AITR) 
# TRAINED ON THREE SYNTHETIC DATASETS (“MisCaption This!” (D3), NESt, AND CHASMA) AND EVALUATED ON VERITE ("TRUE VS. MC"). 
for data_name, data_version in [
    ("newsclippings", "_llava_v1_3"), 
    ("NESt", ""), 
    ("Misalign", "")]:
    
    # Experiments for DT-Transformer and RED-DOT
    run_experiment(data_path=data_path, data_name=data_name, data_name_v=data_version, use_multiclass=False, recon_list=[False], model_version="transformer", transformer_version="default", choose_fusion_method=[["concat_1", "add", "sub", "mul"], ["concat_1"]])
    
    # Experiments for AITR with MUSE. set sim_coding=True and transformer_version="aitr" 
    for pooling_method in ["attention_pooling", "weighted_pooling"]:
        run_experiment(data_path=data_path, data_name=data_name, data_name_v=data_version, sim_coding=True, use_multiclass=False, recon_list=[False], model_version="transformer", transformer_version="aitr", pooling_method=pooling_method)    

    # Ablation experiments for LAMAR. Trained END-TO-END
    run_experiment(data_path=data_path, data_name=data_name, data_name_v=data_version, use_multiclass=False, 
                   recon_list=["integrate_direct", "integrate_gated","integrate_masked","integrate_attention","integrate_image_only", "integrate_text_only"])     



# Experiments for Table 2
# PERFORMANCE OF MODELS ON THE “TRUE VS. OOC” TASK
# TRAINED ON THE NEWSCLIPPINGS DATASET AND EVALUATED ON BOTH NEWSCLIPPINGS AND VERITE

# Experiments for DT-Transformer and RED-DOT
run_experiment(data_path=data_path, use_multiclass="OOC", recon_list=[False], model_version="transformer", transformer_version="default", choose_fusion_method=[["concat_1", "add", "sub", "mul"], ["concat_1"]])

# Experiments for AITR with MUSE
for pooling_method in ["attention_pooling", "weighted_pooling"]:
    run_experiment(data_path=data_path, sim_coding=True, use_multiclass="OOC", recon_list=[False], model_version="transformer", transformer_version="aitr", pooling_method=pooling_method)    

# Experiments for LAMAR 
run_experiment(data_path=data_path, use_multiclass="OOC", recon_list=["integrate_direct", "integrate_gated","integrate_masked","integrate_attention","integrate_image_only", "integrate_text_only"])     


# Experiments for Table 3
# PERFORMANCE OF MODELS TRAINED ON THE MULTI-CLASS “MisCaption This!” (D3) OR THE COMBINED CHASMA AND NEWSCLIPPINGS DATASETS
for data_name, data_version in [
    ("newsclippings", "_llava_v1_3"), 
    ("Misalign", "")
]:
    
    # Experiments for DT-Transformer and RED-DOT
    run_experiment(data_path=data_path, data_name=data_name, data_name_v=data_version, use_multiclass=True, recon_list=[False], model_version="transformer", transformer_version="default", choose_fusion_method=[["concat_1", "add", "sub", "mul"], ["concat_1"]])
    
    # Experiments for AITR with MUSE
    for pooling_method in ["attention_pooling", "weighted_pooling"]:
        run_experiment(data_path=data_path, data_name=data_name, data_name_v=data_version, sim_coding=True, use_multiclass=True, recon_list=[False], model_version="transformer", transformer_version="aitr", pooling_method=pooling_method)    

    # Experiments for LAMAR 
    run_experiment(data_path=data_path, data_name=data_name, data_name_v=data_version, use_multiclass=True, 
                   recon_list=["integrate_direct", "integrate_gated","integrate_masked","integrate_attention","integrate_image_only", "integrate_text_only"])     

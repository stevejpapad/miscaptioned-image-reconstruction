import os
import gc
import json
import time
import torch
import random
import open_clip
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F
from sklearn.utils import resample
from torch.utils.data import DataLoader

def set_seed(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_C(C, pos):
    
    if C == 0:
        return np.zeros(pos.shape[0])    
    else: 
        return np.ones(pos.shape[0])
        
        
def sensitivity_per_class(y_true, y_pred, C):
    
    pos = np.where(y_true == C)[0]
    y_true = y_true[pos]
    y_pred = y_pred[pos]
    
    if C == 2:
        y_true = np.ones(y_true.shape[0]).reshape(-1, 1)
    
    return round((y_pred == y_true).sum() / y_true.shape[0], 4)

def accuracy_CvC(y_true, y_pred, Ca, Cb):
    pos_a = np.where(y_true == Ca)[0]
    pos_b = np.where(y_true == Cb)[0]

    y_pred_a = y_pred[pos_a].flatten()
    y_pred_b = y_pred[pos_b].flatten()   
    
    y_true_a = check_C(Ca, pos_a)
    y_true_b = check_C(Cb, pos_b)
    
    y_pred_avb = np.concatenate([y_pred_a, y_pred_b])
    y_true_avb = np.concatenate([y_true_a, y_true_b])
    
    return round(metrics.accuracy_score(y_true_avb, y_pred_avb), 4)

class DatasetIterator(torch.utils.data.Dataset):

    def __init__(self,input_data,visual_features,textual_features):
        self.input_data = input_data
        self.visual_features = visual_features
        self.textual_features = textual_features

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):

        current = self.input_data.iloc[idx]

        img = self.visual_features[current.image_id].values.astype("float32")
        
        input_txt = self.textual_features[current.id].values.astype("float32")
        
        target_txt = self.textual_features[current.image_id].values.astype("float32")        

        label = float(current.falsified)

        return img, input_txt, label, target_txt

def prepare_dataloader(input_data,visual_features,textual_features,batch_size,num_workers,shuffle=False,):

    dg = DatasetIterator(input_data,visual_features,textual_features)
    dataloader = DataLoader(dg,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True,drop_last=False)

    return dataloader
        

def train_step(model, input_dataloader, use_multiclass, current_epoch, optimizer, chosen_criterion, device, batches_per_epoch):
    
    epoch_start_time = time.time()

    running_loss = 0.0
    running_loss_1 = 0.0
    running_loss_2 = 0.0
    
    reconstruction_loss = 0.0
    clf_loss = 0.0
    
    a = float(chosen_criterion.split(':')[-1]) # Weight to combine the two losses
    
    model.train()
    
    for i, data in enumerate(input_dataloader, 0):

        images = data[0].to(device, non_blocking=True)
        input_texts = data[1].to(device, non_blocking=True).squeeze(1)
        
        target_texts = data[3].to(device, non_blocking=True).squeeze(1) 
        target_texts = F.normalize(target_texts, p=2, dim=1) 
        
        if use_multiclass == True:
            labels = torch.nn.functional.one_hot(data[2].long(), num_classes=3).float().to(device, non_blocking=True)
        else:
            labels = data[2].to(device, non_blocking=True)
                
        optimizer.zero_grad()
                
        output_clf, output_recon = model(images, input_texts)
                
        if "mse" in chosen_criterion:
            reconstruction_loss = F.mse_loss(output_recon, target_texts)
            
        if "bce_clf" in chosen_criterion:            
            if use_multiclass == True:
                clf_loss = F.cross_entropy(output_clf, labels)                 
            else:
                clf_loss = F.binary_cross_entropy_with_logits(output_clf.float(), labels.float()) 
        
        loss = reconstruction_loss + clf_loss * a
                    
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if "mse" in chosen_criterion:
            running_loss_1 += reconstruction_loss.item()
            
        if "bce" in chosen_criterion:       
            running_loss_2 += clf_loss.item()

        print(
            f"[Epoch:{current_epoch + 1}, Batch:{i + 1:5d}/{batches_per_epoch}]. Passed time: {round((time.time() - epoch_start_time) / 60, 1)} minutes. loss: {running_loss / (i+1):.3f}. Reconstruction loss: {round(running_loss_1 / (i+1), 3)}. BCE loss: {round(running_loss_2 / (i+1), 3)}",
            end="\r",
        )  

def eval_step(model, input_dataloader, use_multiclass, current_epoch, chosen_criterion, device, batches_per_epoch):
    
    epoch_start_time = time.time()

    running_loss = 0.0
    running_loss_1 = 0.0
    running_loss_2 = 0.0
    
    reconstruction_loss = 0.0
    clf_loss = 0.0
    
    y_true = []
    y_pred = []
    
    a = float(chosen_criterion.split(':')[-1]) # Weight to combine the two losses
    
    model.eval()
    
    with torch.no_grad():
    
        for i, data in enumerate(input_dataloader, 0):

            images = data[0].to(device, non_blocking=True)
            input_texts = data[1].to(device, non_blocking=True).squeeze(1)
            labels = data[2].to(device, non_blocking=True)            
            
            output_clf, output_recon = model(images, input_texts)
                                                    
            y_pred.extend(output_clf.cpu().detach().numpy())
            y_true.extend(labels.cpu().detach().numpy())
                    
    y_pred = np.vstack(y_pred)    

    if use_multiclass == True:
        y_true = np.vstack(y_true)
        y_true = y_true.flatten()
        
        y_pred_softmax = torch.log_softmax(torch.Tensor(y_pred), dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
        y_pred = y_pred_tags.numpy() 
                
        acc = metrics.accuracy_score(y_true, y_pred)    
        prec = metrics.precision_score(y_true, y_pred, average='macro')
        recall = metrics.recall_score(y_true, y_pred, average='macro') 
        f1 = metrics.f1_score(y_true, y_pred, average='macro')

        results = {
            "epoch": current_epoch,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
        }
        print(results)

    else:
        y_pred = 1/(1 + np.exp(-y_pred))
        y_true = np.array(y_true).reshape(-1,1)
        auc = metrics.roc_auc_score(y_true, y_pred)
        
        y_pred = np.round(y_pred)        
        acc = metrics.accuracy_score(y_true, y_pred)    
        prec = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred) 
        f1 = metrics.f1_score(y_true, y_pred)
        cm = metrics.confusion_matrix(y_true, y_pred, normalize="true").diagonal()
        
        results = {
            "epoch": current_epoch,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            'Pristine': round(cm[0], 4),
            'Falsified': round(cm[1], 4),
        }
        print(results)
    
    return results

def eval_verite(model, verite_data, verite_dataloader, use_multiclass, device, label_map={'true': 0, 'miscaptioned': 1, 'out-of-context': 2}, cur_epoch=-3):
    
    print("\nEvaluation on VERITE")
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for i, data in enumerate(verite_dataloader, 0):

            images = data[0].to(device, non_blocking=True)
            texts = data[1].to(device, non_blocking=True)
            labels = data[2]
            
            output_clf, output_recon = model(images, texts)
            y_pred.extend(output_clf.detach().cpu().numpy())
            y_true.extend(labels.detach().cpu().numpy())


    y_pred = np.vstack(y_pred)
    
    if use_multiclass == True:

        y_pred_softmax = torch.log_softmax(torch.Tensor(y_pred), dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
        y_pred = y_pred_tags.numpy() 
        
        acc = metrics.accuracy_score(y_true, y_pred)   
        matrix = metrics.confusion_matrix(y_true, y_pred)
        cm_results = matrix.diagonal() / matrix.sum(axis=1)

        true_ = cm_results[0]
        miscaptioned_ = cm_results[1]
        out_of_context = cm_results[2]

        verite_results = {
            "epoch": -1,
            "Accuracy": round(acc, 4),
            'True': round(cm_results[0], 4),
            'Miscaptioned': round(cm_results[1], 4),
            'Out-Of-Context': round(cm_results[2], 4)        
        }
        
    else:
        y_true = np.array(y_true).reshape(-1,1)
        y_pred = 1/(1 + np.exp(-y_pred))
    
        y_pred = y_pred.round()
        
        verite_results = {}
    
        verite_results['True'] = sensitivity_per_class(y_true, y_pred, 0)
        verite_results['Miscaptioned'] = sensitivity_per_class(y_true, y_pred, 1)
        verite_results['Out-Of-Context'] = sensitivity_per_class(y_true, y_pred, 2)
    
        verite_results['true_v_miscaptioned'] = accuracy_CvC(y_true, y_pred, 0, 1)
        verite_results['true_v_ooc'] = accuracy_CvC(y_true, y_pred, 0, 2)
        verite_results['miscaptioned_v_ooc'] = accuracy_CvC(y_true, y_pred, 1, 2)
    
        y_true_all = y_true.copy()
        y_true_all[np.where(y_true_all == 2)[0]] = 1
    
        verite_results['accuracy'] = round(metrics.accuracy_score(y_true_all, y_pred), 4)
        verite_results['balanced_accuracy'] = round(metrics.balanced_accuracy_score(y_true_all, y_pred), 4)
    
    print(verite_results)

    return verite_results

def topsis(xM, wV=None):
    m, n = xM.shape

    if wV is None:
        wV = np.ones((1, n)) / n
    else:
        wV = wV / np.sum(wV)

    normal = np.sqrt(np.sum(xM**2, axis=0))

    rM = xM / normal
    tM = rM * wV
    twV = np.max(tM, axis=0)
    tbV = np.min(tM, axis=0)
    dwV = np.sqrt(np.sum((tM - twV) ** 2, axis=1))
    dbV = np.sqrt(np.sum((tM - tbV) ** 2, axis=1))
    swV = dwV / (dwV + dbV)

    arg_sw = np.argsort(swV)[::-1]

    r_sw = swV[arg_sw]

    return np.argsort(swV)[::-1]

def choose_best_model(input_df, metrics, epsilon=1e-6):

    X0 = input_df.copy()
    X0 = X0.reset_index(drop=True)
    X1 = X0[metrics]
    X1 = X1.reset_index(drop=True)
    
    # Stop if the scores are identical in all consecutive epochs
    X1[:-1] = X1[:-1] + epsilon

    if "Accuracy" in metrics:
        X1["Accuracy"] = 1 - X1["Accuracy"]    

    if "Precision" in metrics:
        X1["Precision"] = 1 - X1["Precision"]    

    if "Recall" in metrics:
        X1["Recall"] = 1 - X1["Recall"]          
        
    if "AUC" in metrics:
        X1["AUC"] = 1 - X1["AUC"]
        
    if "F1" in metrics:
        X1["F1"] = 1 - X1["F1"]

    if "Pristine" in metrics:
        X1["Pristine"] = 1 - X1["Pristine"]
        
    if "Falsified" in metrics:
        X1["Falsified"] = 1 - X1["Falsified"]

    if "true_v_miscaptioned" in metrics:
        X1["true_v_miscaptioned"] = 1 - X1["true_v_miscaptioned"]    

    if "true_v_ooc" in metrics:
        X1["true_v_ooc"] = 1 - X1["true_v_ooc"]    

    X_np = X1.to_numpy()
    best_results = topsis(X_np)
    top_K_results = best_results[:1]
    return X0.iloc[top_K_results]

def early_stop(has_not_improved_for, model, optimizer, history, current_epoch, PATH, metrics_list):

    best_index = choose_best_model(
        pd.DataFrame(history), metrics=metrics_list
    ).index[0]
        
    if not os.path.isdir(PATH.split('/')[0]):
        os.mkdir(PATH.split('/')[0])

    if current_epoch == best_index:
        
        print("Checkpoint!\n")
        torch.save(
            {
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            PATH,
        )

        has_not_improved_for = 0
    else:
        
        print("DID NOT CHECKPOINT!\n")
        has_not_improved_for += 1
            
    return has_not_improved_for

def save_results_csv(output_folder_, output_file_, model_performance_):
    print("Save Results ", end=" ... ")
    exp_results_pd = pd.DataFrame(pd.Series(model_performance_)).transpose()
    if not os.path.isfile(output_folder_ + "/" + output_file_ + ".csv"):
        exp_results_pd.to_csv(
            output_folder_ + "/" + output_file_ + ".csv",
            header=True,
            index=False,
            columns=list(model_performance_.keys()),
        )
    else:
        exp_results_pd.to_csv(
            output_folder_ + "/" + output_file_ + ".csv",
            mode="a",
            header=False,
            index=False,
            columns=list(model_performance_.keys()),
        )
    print("Done\n")

def len_df(df, threshold):

    df['caption_len'] = df.caption.str.len()
    
    df = df.pivot(index="image_id", columns="falsified", values=["caption", "caption_len"]).reset_index()
    
    df.columns = ["image_id", "true_caption", "generated_caption", "true_caption_len", "generated_caption_len"]
    
    df["text_length_difference"] = abs(df["true_caption_len"] - df["generated_caption_len"])
    
    filtered_df = df[df["text_length_difference"] <= threshold].drop(columns=["text_length_difference"])
    
    filtered_df = filtered_df.melt(id_vars=["image_id"], 
                             value_vars=["generated_caption", "true_caption"], 
                             var_name="falsified", 
                             value_name="caption")
    
    filtered_df.falsified = filtered_df.falsified.map({'generated_caption': True, 'true_caption': False})
    filtered_df["id"] = filtered_df["image_id"]
    filtered_df["id"] = np.where(filtered_df["falsified"], filtered_df["image_id"].astype(str) + "_fake", filtered_df["image_id"].astype(str))
    
    filtered_df = filtered_df.sample(frac=1, random_state=0)

    return filtered_df

def downsample(df):
    min_count = df['falsified'].value_counts().min()
    
    # Downsample each class to the minimum count
    df_downsampled = pd.concat([
        resample(group, 
                 replace=False, 
                 n_samples=min_count, 
                 random_state=42)
        for label, group in df.groupby('falsified')
    ])
    
    df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)    
    return df_downsampled



def load_data(data_path, data_name, data_name_v, encoder, encoder_version, filter_data, use_multiclass):
    
    if data_name == "newsclippings":
    
        train_generated_data = pd.read_csv(data_path+"miscaptioned/" + data_name + "_train" + data_name_v + ".csv", index_col=0)
        valid_generated_data = pd.read_csv(data_path+"miscaptioned/" + data_name + "_valid" + data_name_v + ".csv", index_col=0)
        test_generated_data = pd.read_csv(data_path+"miscaptioned/" + data_name + "_test" + data_name_v + ".csv", index_col=0)
                        
        train_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/train.json"))
        valid_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/val.json"))
        test_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/test.json"))
    
        train_data = pd.DataFrame(train_data["annotations"])
        valid_data = pd.DataFrame(valid_data["annotations"])
        test_data = pd.DataFrame(test_data["annotations"])
    
        vn_data = json.load(open(data_path + 'VisualNews/origin/data.json'))
        vn_data = pd.DataFrame(vn_data)
        
        vn_data['image_id'] = vn_data['id']
        
        train_data = train_data.merge(vn_data[["caption", "image_id", "image_path"]], on="image_id")
        valid_data = valid_data.merge(vn_data[["caption", "image_id", "image_path"]], on="image_id")
        test_data = test_data.merge(vn_data[["caption", "image_id", "image_path"]], on="image_id")    
        
        # Keep the truthful examples from NewsCLIPpings        
        train_true_data = train_data[train_data.falsified==False]
        valid_true_data = valid_data[valid_data.falsified==False]
        test_true_data = test_data[test_data.falsified==False]
        
        train_ooc_data = train_data[train_data.falsified==True]
        valid_ooc_data = valid_data[valid_data.falsified==True]
        test_ooc_data = test_data[test_data.falsified==True]

        if data_name_v:        
            train_data = pd.concat([train_true_data, train_generated_data])
            valid_data = pd.concat([valid_true_data, valid_generated_data])
            test_data = pd.concat([test_true_data, test_generated_data])    
    
    elif data_name == "NESt":
                
        data_name_v = "vitl14" 
        
        suffix = "_".join([str(x).replace(".", "") for x in [0.5,0.0,0.5]])
    
        train_data = pd.read_csv(data_path+'miscaptioned/train_manipulated_' + data_name + "_" + "vitl14" + '_static_' + suffix +'.csv', index_col=0)
        valid_data = pd.read_csv(data_path+'miscaptioned/valid_manipulated_' + data_name + "_" + "vitl14" + '_static_' + suffix +'.csv', index_col=0)
        test_data = pd.read_csv(data_path+'miscaptioned/test_manipulated_' + data_name + "_" + "vitl14" + '_static_' + suffix + '.csv', index_col=0)
        
        train_data['caption'] = train_data['input_text']
        valid_data['caption'] = valid_data['input_text']
        test_data['caption'] = test_data['input_text']
        
        train_data['image_id'] = train_data['id']
        valid_data['image_id'] = valid_data['id']
        test_data['image_id'] = test_data['id']
    
    elif data_name == "Misalign":
        
        train_data = pd.read_csv(data_path+"miscaptioned/"  + "train_" + data_name + ".csv", index_col=0)
        valid_data = pd.read_csv(data_path+"miscaptioned/"  + "valid_" + data_name + ".csv", index_col=0)
        test_data = pd.read_csv(data_path+"miscaptioned/"  + "test_" + data_name + ".csv", index_col=0)

        train_data_false = train_data[train_data.falsified == True].drop_duplicates('id')
        valid_data_false = valid_data[valid_data.falsified == True].drop_duplicates('id')
        test_data_false = test_data[test_data.falsified == True].drop_duplicates('id')

        train_data_true = train_data[train_data.falsified == False]
        valid_data_true = valid_data[valid_data.falsified == False]
        test_data_true = test_data[test_data.falsified == False]    

        train_data_true = train_data_true[train_data_true.image_id.isin(train_data_false.image_id)]
        valid_data_true = valid_data_true[valid_data_true.image_id.isin(valid_data_false.image_id)]
        test_data_true = test_data_true[test_data_true.image_id.isin(test_data_false.image_id)]

        train_data = pd.concat([train_data_true, train_data_false])
        valid_data = pd.concat([valid_data_true, valid_data_false])
        test_data = pd.concat([test_data_true, test_data_false])   

        train_data = train_data[~train_data.caption.isna()]    
        valid_data = valid_data[~valid_data.caption.isna()]   
        test_data = test_data[~test_data.caption.isna()]    
    else:
        raise("Dataset is not available!")

                
    train_data[['id', 'image_id']] = train_data[['id', 'image_id']].astype('str')
    valid_data[['id', 'image_id']] = valid_data[['id', 'image_id']].astype('str')
    test_data[['id', 'image_id']] = test_data[['id', 'image_id']].astype('str') 
    
    train_data = train_data.sample(frac=1, random_state=0).reset_index(drop=True)
            
    image_embeddings = np.load(data_path + "miscaptioned/" + data_name + data_name_v + '_' + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy").astype('float32') 
    text_embeddings = np.load(data_path + "miscaptioned/" + data_name + data_name_v + '_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy").astype('float32') 
    image_ids = np.load(data_path + "miscaptioned/" + data_name + data_name_v + "_image_ids_" + encoder_version +".npy")
    text_ids = np.load(data_path + "miscaptioned/" + data_name + data_name_v + "_text_ids_" + encoder_version +".npy")
    
    image_embeddings = pd.DataFrame(image_embeddings, index=image_ids).T
    text_embeddings = pd.DataFrame(text_embeddings, index=text_ids).T
    
    image_embeddings.columns = image_embeddings.columns.astype('str')
    text_embeddings.columns = text_embeddings.columns.astype('str')  

    if "Misalign" in data_name and not use_multiclass:
        text_embeddings = text_embeddings.loc[:, ~text_embeddings.columns.duplicated()]
    
    label_map={'true': 0, 'miscaptioned': 1, 'out-of-context': 2}
    
    verite_test = pd.read_csv(data_path + 'VERITE/VERITE.csv', index_col=0)
    verite_test = verite_test.reset_index().rename({'index': 'id', 'label': 'falsified'}, axis=1)
    verite_test['image_id'] = verite_test['id']
    
    verite_test["falsified"] = verite_test["falsified"].map(label_map).astype(int)
    
    verite_test.id = verite_test.id.astype("str")
    verite_test.image_id = verite_test.image_id.astype("str")
    
    verite_text_embeddings = np.load(data_path + "VERITE/VERITE_" + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy").astype('float32')
    verite_image_embeddings = np.load(data_path + "VERITE/VERITE_" + encoder.lower() +"_image_embeddings_" + encoder_version + ".npy").astype('float32')
    verite_image_embeddings = pd.DataFrame(verite_image_embeddings, index=verite_test.id.values).T
    verite_text_embeddings = pd.DataFrame(verite_text_embeddings, index=verite_test.id.values).T
        
    if "filter_len" in filter_data:

        filter_threshold = int(filter_data.split("_")[-1])

        train_data = len_df(train_data, threshold=filter_threshold)
        valid_data = len_df(valid_data, threshold=filter_threshold)
        test_data = len_df(test_data, threshold=filter_threshold)
                    
    train_data[['id', 'image_id']] = train_data[['id', 'image_id']].astype('str')
    valid_data[['id', 'image_id']] = valid_data[['id', 'image_id']].astype('str')
    test_data[['id', 'image_id']] = test_data[['id', 'image_id']].astype('str')         
    
    if use_multiclass == True and data_name == "newsclippings":

        if "filter_len" in filter_data:
            train_data["falsified"] = train_data["falsified"].map({False: 0, True: 1}).astype(int)
            valid_data["falsified"] = valid_data["falsified"].map({False: 0, True: 1}).astype(int)
            test_data["falsified"] = test_data["falsified"].map({False: 0, True: 1}).astype(int)

        else:
            train_data["falsified"] = train_data["falsified"].map({False: 0, "Generated": 1}).astype(int)
            valid_data["falsified"] = valid_data["falsified"].map({False: 0, "Generated": 1}).astype(int)
            test_data["falsified"] = test_data["falsified"].map({False: 0, "Generated": 1}).astype(int)
        
        train_ooc_data["falsified"] = train_ooc_data["falsified"].map({True: 2}).astype(int)
        valid_ooc_data["falsified"] = valid_ooc_data["falsified"].map({True: 2}).astype(int)
        test_ooc_data["falsified"] = test_ooc_data["falsified"].map({True: 2}).astype(int)
    
        train_ooc_data[['id', 'image_id']] = train_ooc_data[['id', 'image_id']].astype('str')
        valid_ooc_data[['id', 'image_id']] = valid_ooc_data[['id', 'image_id']].astype('str')
        test_ooc_data[['id', 'image_id']] = test_ooc_data[['id', 'image_id']].astype('str')     
    
        train_data = pd.concat([train_data, train_ooc_data[train_ooc_data.id.isin(train_data.id.tolist())]])
        valid_data = pd.concat([valid_data, valid_ooc_data[valid_ooc_data.id.isin(valid_data.id.tolist())]])
        test_data = pd.concat([test_data, test_ooc_data[test_ooc_data.id.isin(test_data.id.tolist())]])
    
        train_data = train_data.sample(frac=1, random_state=0)
    
    elif use_multiclass == True and data_name not in ["NESt", "Misalign"]:
    
        train_data_ooc = pd.read_csv(data_path + "miscaptioned/" + data_name + "_train" + data_name_v + "_ooc.csv", index_col=0)
        valid_data_ooc = pd.read_csv(data_path + "miscaptioned/" + data_name + "_valid" + data_name_v + "_ooc.csv", index_col=0)
        test_data_ooc = pd.read_csv(data_path + "miscaptioned/" + data_name + "_test" + data_name_v + "_ooc.csv", index_col=0)
        
        train_data_ooc["falsified"] = train_data_ooc["falsified"].map({True: 2}).astype(int)
        valid_data_ooc["falsified"] = valid_data_ooc["falsified"].map({True: 2}).astype(int)
        test_data_ooc["falsified"] = test_data_ooc["falsified"].map({True: 2}).astype(int)
    
        train_data_ooc[['id', 'image_id']] = train_data_ooc[['id', 'image_id']].astype('str')
        valid_data_ooc[['id', 'image_id']] = valid_data_ooc[['id', 'image_id']].astype('str')
        test_data_ooc[['id', 'image_id']] = test_data_ooc[['id', 'image_id']].astype('str')     
    
        train_data = pd.concat([train_data, train_data_ooc[train_data_ooc.id.isin(train_data.id.tolist())]])
        valid_data = pd.concat([valid_data, valid_data_ooc[valid_data_ooc.id.isin(valid_data.id.tolist())]])
        test_data = pd.concat([test_data, test_data_ooc[test_data_ooc.id.isin(test_data.id.tolist())]])
        
        train_data = train_data.sample(frac=1, random_state=0)

    elif use_multiclass == True and data_name == "Misalign":    

        train_data_ooc = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/train.json"))
        valid_data_ooc = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/val.json"))
        test_data_ooc = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/test.json"))
    
        train_data_ooc = pd.DataFrame(train_data_ooc["annotations"])
        valid_data_ooc = pd.DataFrame(valid_data_ooc["annotations"])
        test_data_ooc = pd.DataFrame(test_data_ooc["annotations"])    

        train_data_ooc[['id', 'image_id']] = train_data_ooc[['id', 'image_id']].astype('str')
        valid_data_ooc[['id', 'image_id']] = valid_data_ooc[['id', 'image_id']].astype('str')
        test_data_ooc[['id', 'image_id']] = test_data_ooc[['id', 'image_id']].astype('str')  

        train_data_ooc["falsified"] = train_data_ooc["falsified"].map({False:0, True: 2}).astype(int)
        valid_data_ooc["falsified"] = valid_data_ooc["falsified"].map({False:0, True: 2}).astype(int)
        test_data_ooc["falsified"] = test_data_ooc["falsified"].map({False:0, True: 2}).astype(int)            
                    
        train_data = pd.concat([train_data, train_data_ooc])
        valid_data = pd.concat([valid_data, valid_data_ooc])
        test_data = pd.concat([test_data, test_data_ooc])
    
        train_data = train_data[train_data.id != 111288]    

        # Only keeps the NewsCLIPpings out-of-context samples
        image_embeddings_nc = np.load(data_path + "miscaptioned/newsclippings"+ "_llava_v1_3" + '_' + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy").astype('float32') 
        text_embeddings_nc = np.load(data_path + "miscaptioned/newsclippings" + "_llava_v1_3" + '_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy").astype('float32') 
        image_ids_nc = np.load(data_path + "miscaptioned/newsclippings" + "_llava_v1_3" + "_image_ids_" + encoder_version +".npy")
        text_ids_nc = np.load(data_path + "miscaptioned/newsclippings" + "_llava_v1_3" + "_text_ids_" + encoder_version +".npy")

        image_embeddings_nc = pd.DataFrame(image_embeddings_nc, index=image_ids_nc).T
        text_embeddings_nc = pd.DataFrame(text_embeddings_nc, index=text_ids_nc).T
        
        image_embeddings_nc.columns = image_embeddings_nc.columns.astype('str')
        text_embeddings_nc.columns = text_embeddings_nc.columns.astype('str')              

        image_embeddings = pd.concat([image_embeddings, image_embeddings_nc], axis=1)
        text_embeddings = pd.concat([text_embeddings, text_embeddings_nc], axis=1)

        image_embeddings = image_embeddings.loc[:, ~image_embeddings.columns.duplicated()]     
        text_embeddings = text_embeddings.loc[:, ~text_embeddings.columns.duplicated()]        

        train_data = downsample(train_data)
        valid_data = downsample(valid_data)  
        test_data = downsample(test_data)        
    
    elif data_name in ["NESt", "Misalign"]:
        pass
    
    else:

        if use_multiclass == "OOC":

            train_data = pd.concat([train_true_data, train_ooc_data])
            valid_data = pd.concat([valid_true_data, valid_ooc_data])
            test_data = pd.concat([test_true_data, test_ooc_data])     

            train_data[['id', 'image_id']] = train_data[['id', 'image_id']].astype('str')
            valid_data[['id', 'image_id']] = valid_data[['id', 'image_id']].astype('str')
            test_data[['id', 'image_id']] = test_data[['id', 'image_id']].astype('str')             

            train_data = train_data.sample(frac=1, random_state=0)

            train_data["falsified"] = train_data["falsified"].map({False: 0, True: 1}).astype(int)
            valid_data["falsified"] = valid_data["falsified"].map({False: 0, True: 1}).astype(int)
            test_data["falsified"] = test_data["falsified"].map({False: 0, True: 1}).astype(int)        

        else:
        
            if "filter_len" in filter_data:
                train_data["falsified"] = train_data["falsified"].map({False: 0, True: 1}).astype(int)
                valid_data["falsified"] = valid_data["falsified"].map({False: 0, True: 1}).astype(int)
                test_data["falsified"] = test_data["falsified"].map({False: 0, True: 1}).astype(int)
    
            else:
                train_data["falsified"] = train_data["falsified"].map({False: 0, "Generated": 1}).astype(int)
                valid_data["falsified"] = valid_data["falsified"].map({False: 0, "Generated": 1}).astype(int)
                test_data["falsified"] = test_data["falsified"].map({False: 0, "Generated": 1}).astype(int)
    
    display(train_data.falsified.value_counts(), train_data.shape)
    
    return train_data, valid_data, test_data, verite_test, image_embeddings, text_embeddings, verite_image_embeddings, verite_text_embeddings


class ImageIteratorSource(torch.utils.data.Dataset):
    def __init__(
        self,
        input_data,
        images_path,
        vis_processors,
    ):
        self.input_data = input_data
        self.images_path = images_path
        self.vis_processors = vis_processors

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        current = self.input_data.iloc[idx]
        
        img_path = self.images_path + current.image_path.split('./')[-1]        
        image = Image.open(img_path).convert('RGBA')
        img = self.vis_processors(image)    
        idx = current.image_id
        
        return idx, img


class TextIteratorSource(torch.utils.data.Dataset):
    def __init__(
        self,
        input_data,
        txt_processors,
    ):
        self.input_data = input_data
        self.txt_processors = txt_processors

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        current = self.input_data.iloc[idx]

        txt = self.txt_processors(current.caption)                    
        idx = str(current.id)
        
        return idx, txt

def prepare_source_dataloader(input_data, vis_processors, txt_processors, images_path, batch_size, num_workers, shuffle):
       
    img_dataloader = DataLoader(
        ImageIteratorSource(input_data.drop_duplicates("image_id"), 
                            images_path, 
                            vis_processors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )

    txt_dataloader = DataLoader(
        TextIteratorSource(input_data.drop_duplicates("id"), txt_processors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )    
    
    return img_dataloader, txt_dataloader

def extract_features(data_path, images_path, data_name, data_name_v, encoder="CLIP", encoder_version="ViT-L/14"):

    train_generated_data = pd.read_csv(data_path + "miscaptioned/" + data_name + "_train" + data_name_v + ".csv", index_col=0)
    valid_generated_data = pd.read_csv(data_path + "miscaptioned/" + data_name + "_valid" + data_name_v + ".csv", index_col=0)
    test_generated_data = pd.read_csv(data_path + "miscaptioned/" + data_name + "_test" + data_name_v + ".csv", index_col=0)
    
    train_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/train.json"))
    valid_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/val.json"))
    test_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/test.json"))
    
    train_data = pd.DataFrame(train_data["annotations"])
    valid_data = pd.DataFrame(valid_data["annotations"])
    test_data = pd.DataFrame(test_data["annotations"])
    
    vn_data = json.load(open(data_path + 'VisualNews/origin/data.json'))
    vn_data = pd.DataFrame(vn_data)
    vn_data['image_id'] = vn_data['id']
    
    # To include OOC samples from NewsCLIPpings
    train_data = train_data.merge(vn_data[["id", "caption"]], on="id")    
    valid_data = valid_data.merge(vn_data[["id", "caption"]], on="id")    
    test_data = test_data.merge(vn_data[["id", "caption"]], on="id")
    train_data = train_data.merge(vn_data[["image_id", "image_path"]], on="image_id")
    valid_data = valid_data.merge(vn_data[["image_id","image_path"]], on="image_id")
    test_data = test_data.merge(vn_data[["image_id","image_path"]], on="image_id")

    # Combine NewsCLIPpings with the generated data
    train_data = pd.concat([train_data, train_generated_data])
    valid_data = pd.concat([valid_data, valid_generated_data])
    test_data = pd.concat([test_data, test_generated_data])
    all_data = pd.concat([train_data, valid_data, test_data])
    
    # This step is necessary for Latent Reconstruction -> to have access to the correct texts of OOC images!
    t = all_data[all_data.falsified == True]
    f = all_data[all_data.falsified == False]
    missing = t[~t.image_id.isin(f.id)].image_id
    missing_data = vn_data[vn_data.id.isin(missing)][['id', 'caption', 'image_path', 'image_id']]
    missing_data["falsified"] = False
    all_data = pd.concat([all_data, missing_data])

    # Text pre-processing
    all_data['caption'] = all_data['caption'].str.replace(r'[^\w\s]', ' ', regex=True)
    all_data['caption'] = all_data['caption'].str.lower()

    # Load CLIP
    encoder_version = encoder_version.replace('-', '').replace('/', '') 
    choose_gpu = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu)
    device = torch.device("cuda:"+str(choose_gpu) if torch.cuda.is_available() else "cpu")
    model, _, vis_processors = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    model.to(device)
    txt_processors = open_clip.get_tokenizer('ViT-L-14')

    output_path = data_path + 'miscaptioned/'
    text_ids, image_ids, all_text_features, all_visual_features = [], [], [], []
    
    img_dataloader, txt_dataloader = prepare_source_dataloader(all_data, 
                                                               vis_processors, 
                                                               txt_processors,  
                                                               images_path,
                                                               256, 
                                                               8, 
                                                               False)
    
    if not os.path.isdir(output_path + 'temp_visual_features'):
        os.makedirs(output_path + 'temp_visual_features')

    # Extract the features for a batch of images. Save as numpy 
    print("Extract features from image evidence. Save the batch as a numpy file")
    batch_count = 0
    with torch.no_grad():
    
        for idx, img in tqdm(img_dataloader):
    
            img = img.to(device)
            image_features = model.encode_image(img)
            image_features = image_features.reshape(image_features.shape[0], -1).cpu().detach().numpy()
            
            np.save(output_path + 'temp_visual_features/' + data_name + data_name_v + '_' + encoder.lower() + '_' + str(batch_count), image_features) 
            del image_features
            del img
    
            batch_count += 1
            image_ids.extend(idx)
    
            torch.cuda.empty_cache()
            gc.collect()
    
    print("Save: ", output_path)
    image_ids = np.stack(image_ids)
    np.save(output_path + data_name + data_name_v + "_image_ids_" + encoder_version +"_v222.npy", image_ids)    

    # Loads all feature-batches and creates a numpy file
    print("Load visual features (numpy files) and concatenate into a single file")
    image_embeddings = []
    for batch_count in range(img_dataloader.__len__()):
    
        print(batch_count, end='\r')
        image_features = np.load(output_path + 'temp_visual_features/' + data_name + data_name_v + '_' + encoder.lower() + '_' + str(batch_count) + '.npy') 
        image_embeddings.extend(image_features)
    
    image_embeddings = np.array(image_embeddings)
    np.save(output_path + data_name + data_name_v + '_' + encoder.lower() + "_image_embeddings_" + encoder_version + "_v222.npy", image_embeddings) 

    # Feature extraction for texts
    print("Extract features from texts")
    with torch.no_grad():
        for idx, txt in tqdm(txt_dataloader):
    
            text_features = model.encode_text(txt.squeeze(1).to(device))            
            text_features = text_features.reshape(text_features.shape[0], -1)
            all_text_features.extend(text_features.cpu().detach().numpy())
            text_ids.extend(idx)
        
    all_text_features = np.stack(all_text_features)
    
    np.save(output_path + data_name + data_name_v + '_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy", all_text_features)    
    text_ids = np.stack(text_ids)
    np.save(output_path + data_name + data_name_v + "_text_ids_" + encoder_version +".npy", text_ids)

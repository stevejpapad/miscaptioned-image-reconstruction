import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class mlp(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob=0.1, final_output=False):
        super(mlp, self).__init__()
        
        self.layers_list = nn.ModuleList()

        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(input_size, hidden_size),        
                nn.GELU(),  
            )
            self.layers_list.append(layer)
            
            input_size = hidden_size
        
        if final_output:
            self.layers_list.append(nn.Linear(input_size, final_output)) 
                           
    def forward(self, x):
        
        for layer in self.layers_list:
            x = layer(x)
            
        return x 

class RECONSTRUCTOR(nn.Module):
    def __init__(
        self,
        use_reconstruction,
        recon_fusion_method,
        reconstruction_layers,
        emb_dim=768,
    ):

        super(RECONSTRUCTOR, self).__init__()
                
        self.emb_dim = emb_dim      
        self.use_reconstruction = use_reconstruction
        self.recon_fusion_method = recon_fusion_method   
                            
        self.rec_token = nn.Parameter(torch.randn(self.emb_dim))
        self.rec_token.requires_grad = True      

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=4,
                dim_feedforward=1024,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=False,                
            ),
            num_layers=4,
        )           
                        
    def forward(self, images, texts):

        b_size = images.shape[0]
                
        if "image_only" in self.use_reconstruction: 
            x = images

        elif "text_only" in self.use_reconstruction: 
            x = texts
        
        else:
            x = combine_features(images, texts, self.recon_fusion_method)   
                    
        rec_token = self.rec_token.expand(b_size, 1, -1)      

        x = x.reshape(b_size, -1, self.emb_dim)
        x = torch.cat([rec_token, x], dim=1)
        
        y = self.transformer(x)[:,0,:]              
        
        return y

def combine_features(a, b, fusion_method):

    if "concat_1" in fusion_method:  
        x = torch.cat([a, b], dim=1)

    if 'add' in fusion_method:
        added = torch.add(a, b)
        x = torch.cat([x, added], axis=1)        

    if 'mul' in fusion_method:
        mult = torch.mul(a, b)
        x = torch.cat([x, mult], axis=1)        

    if 'sub' in fusion_method:
        sub = torch.sub(a, b)
        x = torch.cat([x, sub], axis=1)   
        
    return x

class similarity_encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, sims_to_keep, dropout_prob=0.0):
        super(similarity_encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sims_to_keep = sims_to_keep
        
        self.input_size = 1 + len(self.sims_to_keep)-1                 
        self.sim_coder_mlp = mlp(self.input_size, self.hidden_size, self.num_layers)
                                   
    def forward(self, img, txt):
        
        img_txt = F.cosine_similarity(img, txt, dim=1)
        x = img_txt.unsqueeze(1)                            
        x = self.sim_coder_mlp(x)
        return x    

# Used for the large-scale pre-training ablation experiments 
class RECONSTRUCTOR_PT(nn.Module):
    def __init__(
        self,
        use_reconstruction,
        recon_fusion_method,
        reconstruction_layers,   
        emb_dim=768,
    ):

        super(RECONSTRUCTOR_PT, self).__init__()
                
        self.emb_dim = emb_dim      
        self.use_reconstruction = use_reconstruction 
        self.recon_fusion_method = recon_fusion_method          
            
        self.rec_token = nn.Parameter(torch.randn(self.emb_dim))
        self.rec_token.requires_grad = True      

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=4,
                dim_feedforward=1024,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=False,                
            ),
            num_layers=4,
        )     
        
    def forward(self, images, texts):

        b_size = images.shape[0]
            
        if "true_recon_multimodal_gaussian" in self.use_reconstruction:

            # i.e., use_reconstruction = "pretrained_gaussian_0_0.2"
            mean = float(use_reconstruction.split("_")[-2])
            std_dev = float(use_reconstruction.split("_")[-1])
            
            noise = torch.normal(mean, std_dev, size=texts.size()).to(texts.device)
            noisy_texts = texts + noise
            x = combine_features(images, noisy_texts, self.recon_fusion_method)   

        elif "true_recon_multimodal_dropout" in self.use_reconstruction:

            # i.e., use_reconstruction="pretrained_dropout_0.5"
            drop_rate = float(use_reconstruction.split("_")[-1])
            noisy_texts = F.dropout(texts, p=drop_rate, training=True)
            x = combine_features(images, noisy_texts, self.recon_fusion_method)  
                        
        rec_token = self.rec_token.expand(b_size, 1, -1)      

        x = x.reshape(b_size, -1, self.emb_dim)
        x = torch.cat([rec_token, x], dim=1)
        
        y = self.transformer(x)[:,0,:]           
            
        return y

class LAMAR(nn.Module):
    def __init__(self, 
                 emb_dim, 
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
                 dropout_prob=0.1, 
                 activation="gelu"):
        super(LAMAR, self).__init__()

        # Code for the LAMAR architecture but also contains components for reproducing DT-Transformer, RED-DOT, and AITR
        
        clf_input_size = 0
        
        # General parameters
        self.emb_dim = emb_dim
        self.fusion_method = fusion_method
        self.model_version = model_version
        self.transformer_version = transformer_version
        self.pooling_method = pooling_method      
        self.use_multiclass = use_multiclass
                    
        # # Reconstructor
        self.use_reconstruction = use_reconstruction
        self.num_reconstruction_layers = num_reconstruction_layers
        
        # # Similarity Encoding: Only used for AITR
        self.sim_coding = sim_coding
        self.num_sim_coding_layers = num_sim_coding_layers
        self.sims_to_keep = sims_to_keep        
        
        if self.use_reconstruction:

            recon_fusion = self.fusion_method
            
            self.recon_component = RECONSTRUCTOR(
                use_reconstruction,
                recon_fusion,
                self.num_reconstruction_layers,
                self.emb_dim,
                dropout=dropout_prob)
                                    
            clf_input_size = emb_dim * len(fusion_method) + 2 * emb_dim        

            if "gated" in self.use_reconstruction:
                recon_gate_size = emb_dim * len(fusion_method) + emb_dim  
                self.gate_fc = nn.Linear(recon_gate_size, 1)

            elif "masked" in self.use_reconstruction:
                recon_mask_size = emb_dim * len(fusion_method) + emb_dim  
                self.mask_fc = nn.Linear(recon_mask_size, 1)               

            elif "attention" in self.use_reconstruction:

                self.query_fc = nn.Linear(emb_dim, emb_dim)
                self.key_fc = nn.Linear(emb_dim, emb_dim)
                self.value_fc = nn.Linear(emb_dim, emb_dim)             
                
        if self.sim_coding:
            self.sim_encoder_component = similarity_encoder(self.emb_dim, 
                                                       self.num_sim_coding_layers, 
                                                       self.sims_to_keep, 
                                                       dropout_prob=dropout_prob)    
            
            clf_input_size += self.sim_encoder_component.hidden_size           
            
        if not self.use_reconstruction and not self.sim_coding and self.model_version == None:
            clf_input_size = self.emb_dim * len(fusion_method) + self.emb_dim  
                        
        if self.model_version == "transformer" and self.transformer_version == "default":

            # FOR DT-Transformer and RED-DOT (baseline - no external evidence)
            self.tf_head = tf_h_l[0]
            self.tf_layers = len(tf_h_l)
            self.tf_dim = tf_dim
            
            self.cls_token = nn.Parameter(torch.randn(self.emb_dim))
            self.cls_token.requires_grad = True            
            
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.emb_dim,
                    nhead=self.tf_head,
                    dim_feedforward=self.tf_dim,
                    dropout=dropout_prob,
                    activation=activation,
                    batch_first=True,
                    norm_first=True,                
                ),
                num_layers=self.tf_layers,
            )
            
            clf_input_size = emb_dim

        elif self.model_version == "transformer" and self.transformer_version == "aitr":    
            
            self.tf_dim = tf_dim
            transformer_list = nn.ModuleList()

            self.cls_token = nn.Parameter(torch.randn(self.emb_dim))
            self.cls_token.requires_grad = True                 
            
            for num_heads in tf_h_l:
                transformer_layer = nn.TransformerEncoderLayer(
                    d_model=emb_dim,
                    nhead=num_heads,
                    dim_feedforward=self.tf_dim,
                    dropout=dropout_prob,
                    activation=activation,
                    batch_first=True,
                    norm_first=True,
                )
                transformer_list.append(transformer_layer)     

            self.transformer = transformer_list

            if self.pooling_method == "weighted_pooling":                
                self.wp_linear = nn.Linear(emb_dim, emb_dim)
                self.softmax = nn.Softmax(dim=-1)

            elif self.pooling_method == "attention_pooling":
                
                self.q_layer = nn.Linear(emb_dim, emb_dim)
                self.k_layer = nn.Linear(emb_dim, emb_dim)
                self.v_layer = nn.Linear(emb_dim, emb_dim)
                self.softmax = nn.Softmax(dim=-1)  
                
            clf_input_size = emb_dim
        
        # Classifier 
        self.classifier_version = classifier_version

        self.clf_input_size = clf_input_size
   
        num_clf_layers = int(classifier_version.split("_")[-1])

        self.clf = mlp(self.clf_input_size, 
                       emb_dim, 
                       num_clf_layers, 
                       dropout_prob, 
                       final_output=3 if self.use_multiclass==True else 1)     
    
    def forward(self, img, txt):
        
        b_size = img.shape[0]
                            
        recon = None
        
        if self.use_reconstruction:     
            recon = self.recon_component(img, txt)
                            
            if "integrate_gated" in self.use_reconstruction:            
                fused_features = combine_features(img, txt, self.fusion_method)  
                gate = torch.sigmoid(self.gate_fc(fused_features))  
                weighted_recon = gate * recon
                x = torch.cat([fused_features, weighted_recon], dim=1)

            elif self.use_reconstruction == "integrate_masked":

                fused_features = combine_features(img, txt, self.fusion_method)  
                mask_prob = torch.sigmoid(self.mask_fc(fused_features)) 
                mask = torch.bernoulli(mask_prob)                         
                masked_recon = mask * recon
                x = torch.cat([fused_features, masked_recon], dim=1)     

            elif "integrate_attention" in self.use_reconstruction:
                stack_x = torch.stack([img, txt, recon], dim=1)
                
                query = self.query_fc(stack_x) 
                key = self.key_fc(stack_x)      
                value = self.value_fc(stack_x) 
                
                scores = torch.bmm(query, key.transpose(1, 2))
                scores = scores / (self.emb_dim ** 0.5)  
                attention_weights = F.softmax(scores, dim=-1) 
                attended_inputs = torch.bmm(attention_weights, value)
                
                aggregated_recon  = attended_inputs.mean(dim=1) 
                fused_features = combine_features(img, txt, self.fusion_method)                  
                x = torch.cat([fused_features, aggregated_recon], dim=1) 
        
            else:                         
                fused_features = combine_features(img, txt, self.fusion_method)               
                x = torch.cat([fused_features,recon], dim=1)        
            
        else:
            x = combine_features(img, txt, self.fusion_method) 

        if self.sim_coding:
            sim_enc = self.sim_encoder_component(img, txt)  
            x = torch.cat([x, sim_enc], dim=1)

        if self.model_version == "transformer":
            
            cls_token = self.cls_token.expand(b_size, 1, -1) 
            
            fused_features = combine_features(img, txt, self.fusion_method) 
            fused_features = fused_features.reshape(b_size, -1, self.emb_dim)

            if self.use_reconstruction and self.sim_coding:
                x = torch.cat([cls_token, fused_features, recon.unsqueeze(1), sim_enc.unsqueeze(1)], dim=1)
            
            elif self.sim_coding:
                x = torch.cat([cls_token, fused_features, sim_enc.unsqueeze(1)], dim=1)
                        
            else:
                x = torch.cat([cls_token, fused_features], dim=1)    
        
            if self.transformer_version == "default":
                x = self.transformer(x)[:,0,:]

            if self.transformer_version == "aitr":
                        
                outputs = []
                for layer in self.transformer:
                    x = layer(x)                            
                    outputs.append(x[:,0,:])
                
                x_t = torch.stack(outputs, dim=1)  
                            
                if self.pooling_method == "weighted_pooling":
                    
                    w = self.softmax(self.wp_linear(x_t))
                    x = torch.sum(w * x_t, dim=1)         

                elif self.pooling_method == "attention_pooling":                   
                    
                    Q = self.q_layer(x_t)  
                    K = self.k_layer(x_t)     
                    V = self.v_layer(x_t)
                    
                    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5) 
                    attention_weights = self.softmax(attention_scores)
                    attention_output = torch.matmul(attention_weights, V) 

                    x = attention_output.mean(1)

                else:
                    raise("Choose between weighted and attention pooling for AITR!")

        if self.use_multiclass==True:
            y = self.clf(x)
        else:            
            y = self.clf(x).flatten()
        
        return y, recon
